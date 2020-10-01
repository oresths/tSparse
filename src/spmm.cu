#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>

#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "cusp/timer.h"

#include <thrust/tuple.h>
//#include <thrust/device_ptr.h>
//#include <thrust/device_malloc.h>
//#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/functional.h> //bit_or
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/omp/execution_policy.h>

#include <algorithm> //find
#include <vector>

#include "mm.h"


__global__ void warmup(float4 *output)
{
    const unsigned int tid = threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x+threadIdx.x;

    float4 value = make_float4(threadIdx.z/threadIdx.x, threadIdx.z*threadIdx.z, 3, threadIdx.y+threadIdx.x);
    output[tid] = value;
}

// I made this function because in CUSP in GPU version they don't compact the output. In CPU they do.
template <typename MatrixType>
void cusp_multiplyGPU(const MatrixType& A, const MatrixType& B, MatrixType& C)
{
    using thrust::system::detail::generic::select_system;

    typename MatrixType::memory_space System;
    typedef typename MatrixType::value_type ValueType;

    auto exec = thrust::detail::derived_cast(thrust::detail::strip_const(System));

    cusp::multiply(A, B, C);

    int num_zeros = thrust::count(exec, C.values.begin(), C.values.end(), ValueType(0));

    // The result of the elementwise operation contains zero entries so we need
    // to contract the result to produce a strictly valid COO matrix
    if(num_zeros != 0)
    {
        int num_reduced_entries =
            thrust::remove_if(exec,
                thrust::make_zip_iterator(
                  thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin(), C.values.begin())),
                thrust::make_zip_iterator(
                  thrust::make_tuple(C.row_indices.end(),   C.column_indices.end(), C.values.end())),
                C.values.begin(),
                thrust::placeholders::_1 == ValueType(0)) -
            thrust::make_zip_iterator(
                thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin(), C.values.begin()));

        C.resize(C.num_rows, C.num_cols, num_reduced_entries);
    }
}

struct find_tile_index
{
    using IndexType = int;
    using LongIndexType = uint64_t;
    using BMPType = uint64_t;

    IndexType ncols_;

    find_tile_index(IndexType num_cols) {
        ncols_ = num_cols / BMP_DIM + ((num_cols % BMP_DIM)?1:0) ;
    }

    __host__ __device__
    void operator()(thrust::tuple<IndexType &, IndexType &, LongIndexType &, BMPType &> x) {
        LongIndexType ncols = ncols_; //promotes result to uint64_t, necessary for bigger matrices
        // Absolute index of the tile this element belongs to
        x.get<2>() = (x.get<0>() / BMP_DIM) * ncols + x.get<1>() / BMP_DIM;

        //Absolute index of element inside its tile
        x.get<3>() = 1ULL << ( (x.get<0>() % BMP_DIM) * BMP_DIM + x.get<1>() % BMP_DIM );
    }
};

struct absolute2relative
{
    using IndexType = int;
    using LongIndexType = uint64_t;

    IndexType ncols_;

    absolute2relative(IndexType num_cols): ncols_(num_cols) {}

    __host__ __device__
    void operator()(thrust::tuple<LongIndexType &, IndexType &, IndexType &> x) {
        LongIndexType ncols = ncols_; //promotes result to uint64_t, necessary for bigger matrices
        x.get<1>() = x.get<0>() / ncols;
        x.get<2>() = x.get<0>() - x.get<1>() * ncols; //modulo
    }
};

struct absolute
{
    using ValueType = float;

  __host__ __device__
  void operator()(ValueType &x)
  {
      x = fabsf(x);
  }
};

struct bmp_popcount_d
{
    using UnsignedIndexType = uint32_t;
    using BMPType = uint64_t;

  __device__
  UnsignedIndexType operator()(BMPType rhs)
  {
      return (UnsignedIndexType) __popcll(rhs); //TODO not GPU compatible
  }
};


int coo2bmp_noTuple_d(const cusp::coo_matrix<int, float, cusp::device_memory>& in,
    cusp::coo_matrix<int, uint64_t, cusp::device_memory>& out,
    thrust::device_vector<float>& elems, thrust::device_vector<uint32_t>& idx) {

    using IndexType = int;
    using ElemIndexType = uint32_t;
    using UnsignedIndexType = uint32_t;
    using LongIndexType = uint64_t;
    using BMPType = uint64_t;
    using ValueType = float;
    using ValueTypeBMP = uint64_t;
    using COOHostBMP = cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::host_memory>;
    using COODevBMP =  cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::device_memory>;
    using COOHost =    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>;
    using COODev =     cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>;

    auto exec = thrust::cuda::par;

    COODev in_copy(in);

    // sort COO first, it is needed for elem_array. The COO matrix gets sorted by row, and each row by column.
    thrust::sort_by_key(exec, in_copy.column_indices.begin(), in_copy.column_indices.end(), thrust::make_zip_iterator(
            thrust::make_tuple(in_copy.row_indices.begin(), in_copy.values.begin())));
    thrust::stable_sort_by_key(exec, in_copy.row_indices.begin(), in_copy.row_indices.end(), thrust::make_zip_iterator(
            thrust::make_tuple(in_copy.column_indices.begin(), in_copy.values.begin())));

    thrust::device_vector<LongIndexType> tile_indices(in_copy.num_entries); //Absolute index of the tile each element belongs to
    thrust::device_vector<BMPType> position(in_copy.num_entries); //Absolute index of each element inside respective tile (1<<index)

    // Finds 2 things. a) In which tile each element belongs. Tile is returned with absolute indexing. b) What is the
    // position of each element in the respective tile. The position is returned with absolute indexing.
    thrust::for_each(exec,
            thrust::make_zip_iterator(
                    thrust::make_tuple(in_copy.row_indices.begin(), in_copy.column_indices.begin(), tile_indices.begin(),
                            position.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(in_copy.row_indices.end(), in_copy.column_indices.end(), tile_indices.end(), position.end())),
            find_tile_index(in_copy.num_cols));

    // Sort row_indices, col_indices, values and positions in tile by the absolute index of the tile. The sort is stable
    // in order to keep the order of values (elements). The values are expected to come from a COO matrix that has the
    // rows ordered and the columns of each row ordered.
    thrust::stable_sort_by_key(exec, tile_indices.begin(), tile_indices.end(),
            thrust::make_zip_iterator(
                    thrust::make_tuple(in_copy.row_indices.begin(), in_copy.column_indices.begin(), in_copy.values.begin(),
                            position.begin())));

    thrust::device_vector<LongIndexType> tile_indices_unique(in_copy.num_entries); //Unique absolute indices of tiles
    thrust::device_vector<BMPType> bmp(in_copy.num_entries);

    thrust::equal_to<UnsignedIndexType> binary_pred;
    thrust::bit_or<BMPType> binary_op;
    // Elements are reduced based on the index of the tile they belong to. This function returns the unique tile indices and the
    // the result of reduction is the total bmp of all elements that belong to the same tile.
    auto new_end = thrust::reduce_by_key(exec, tile_indices.begin(), tile_indices.end(), position.begin(), tile_indices_unique.begin(),
            bmp.begin(), binary_pred, binary_op);

    UnsignedIndexType num_of_tiles = new_end.first - tile_indices_unique.begin();

    idx.resize(num_of_tiles);

    // transform BMP to population counts
    thrust::transform(exec, bmp.begin(), new_end.second, idx.begin(), bmp_popcount_d());

    // convert population counts to offsets
    thrust::exclusive_scan(exec, idx.begin(), idx.end(), idx.begin(), UnsignedIndexType(0));

    out.num_rows = in_copy.num_rows / BMP_DIM  + ((in_copy.num_rows % BMP_DIM)?1:0) ;
    out.num_cols = in_copy.num_cols / BMP_DIM  + ((in_copy.num_cols % BMP_DIM)?1:0) ;
    out.num_entries = num_of_tiles;
    out.resize(out.num_rows, out.num_cols, out.num_entries);

    // Convert absolute tile indices to relative indexing, to be stored in the COO matrix of the output
    thrust::for_each(exec,
            thrust::make_zip_iterator(
                    thrust::make_tuple(tile_indices_unique.begin(), out.row_indices.begin(), out.column_indices.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(new_end.first, out.row_indices.end(), out.column_indices.end())),
            absolute2relative(out.num_cols));

    thrust::copy(bmp.begin(), new_end.second, out.values.begin());

    elems.resize(in_copy.num_entries);
    thrust::copy(in_copy.values.begin(), in_copy.values.end(), elems.begin());

    return 1;
}


struct write_values_noTuple_d
{
    using IndexType = int;
    using ElemIndexType = uint32_t;
    using UnsignedIndexType = uint32_t;
    using BMPType = uint64_t;
    using ValueType = float;
    using ValueTypeBMP = uint64_t;
    using COODev =    cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>;
    using ElementVector = thrust::device_vector<ValueType>;

    ValueType * elems_;
    IndexType * row_indices_;
    IndexType * column_indices_;
    ValueType * values_;

    write_values_noTuple_d(ValueType * elems, IndexType * row_indices, IndexType * column_indices, ValueType * values) :
            elems_(elems), row_indices_(row_indices), column_indices_(column_indices), values_(values) {
    }

    __device__
    void operator()(
            const thrust::tuple<const IndexType &, const IndexType &, const ValueTypeBMP &, const ElemIndexType &> x) {

        ElemIndexType idx = x.get<3>();
        uint64_t bmp = x.get<2>();
        uint64_t last_digit = 1;
        for (int i = 0; i < BMP_DIM*BMP_DIM; ++i) {
            if (bmp & last_digit) {
                row_indices_[idx]    = x.get<0>() * BMP_DIM + i / BMP_DIM;
                column_indices_[idx] = x.get<1>() * BMP_DIM + i % BMP_DIM;
                values_[idx] = elems_[idx];
                idx++;
            }
            bmp >>= 1;
        }
    }
};


// Output matrix should have been sized (num_rows,num_cols,num_entries) before calling this function.
// num_rows,num_cols should be from the input. num_entries should be the size of the elem array of the result.
int bmp2coo_noTuple_d(const cusp::coo_matrix<int, uint64_t, cusp::device_memory>& in,
        thrust::device_vector<float>& elems, thrust::device_vector<uint32_t>& idx,
        cusp::coo_matrix<int, float, cusp::device_memory>& out) {

    using IndexType = int;
    using ElemIndexType = uint32_t;
    using UnsignedIndexType = uint32_t;
    using LongIndexType = uint64_t;
    using BMPType = uint64_t;
    using ValueType = float;
    using ValueTypeBMP = uint64_t;
    using COOHostBMP = cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::host_memory>;
    using COODevBMP =  cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::device_memory>;
    using COOHost =    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>;
    using COODev =     cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>;

    auto exec = thrust::cuda::par;

    IndexType * raw_out_row_indices =  thrust::raw_pointer_cast(&out.row_indices[0]);
    IndexType * raw_out_column_indices =  thrust::raw_pointer_cast(&out.column_indices[0]);
    ValueType * raw_out_tiles =  thrust::raw_pointer_cast(&out.values[0]);

    ValueType *raw_elems = thrust::raw_pointer_cast(&elems[0]);

    thrust::for_each(exec,
            thrust::make_zip_iterator(
                    thrust::make_tuple(in.row_indices.begin(), in.column_indices.begin(), in.values.begin(), idx.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(in.row_indices.end(), in.column_indices.end(), in.values.end(), idx.end())),
            write_values_noTuple_d(raw_elems, raw_out_row_indices, raw_out_column_indices, raw_out_tiles));


    // sort COO to be CUSP compatible (although not that much sorting is needed).
    // The COO matrix gets sorted by row, and each row by column.
    thrust::sort_by_key(out.column_indices.begin(), out.column_indices.end(), thrust::make_zip_iterator(
            thrust::make_tuple(out.row_indices.begin(), out.values.begin())));
    thrust::stable_sort_by_key(out.row_indices.begin(), out.row_indices.end(), thrust::make_zip_iterator(
            thrust::make_tuple(out.column_indices.begin(), out.values.begin())));


    return 1;
}

template <typename T>
struct greater_equal_absf
{
    const T threshold_;

    greater_equal_absf(const T threshold): threshold_(threshold) {}

  __host__ __device__
  bool operator()(const T &x)
  {
    return fabsf(x) >= threshold_;
  }
};

template <typename T>
struct lesser_equal_absf
{
    const T threshold_;

    lesser_equal_absf(const T threshold): threshold_(threshold) {}

  __host__ __device__
  bool operator()(const T &x)
  {
    return fabsf(x) <= threshold_;
  }
};

// StrictWeakOrdering
struct tuple_order
{
    using IndexType = int;
    using DataType = thrust::tuple<IndexType &, IndexType &>;

  __host__ __device__
  bool operator()(DataType lhs, DataType rhs)
  {
      bool less_than = false;

      if (lhs.get<0>() < rhs.get<0>())
          less_than = true;
      else if (lhs.get<0>() == rhs.get<0>())
          if (lhs.get<1>() < rhs.get<1>())
              less_than = true;

      return less_than;
  }
};

template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return fabsf(b - a);
    }
};

template <typename T>
struct abs_diff_ratio : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        // add a small value to denominator for when comparing very small numbers
        return fabsf(b - a) / (fabsf(a) + 0.1f);
    }
};

struct abs_diffDouble
{
    using ValueType = float;

    __host__ __device__
    double operator()(const ValueType& a, const ValueType& b)
    {
        return fabs((double)b - (double)a);
    }
};

struct abs_diff_ratioDouble
{
    using ValueType = float;

    __host__ __device__
    double operator()(const ValueType& a, const ValueType& b)
    {
        // add a small value to denominator for when comparing very small numbers
        return fabs((double)b - (double)a) / (double)(fabs(a) + 0.1);
    }
};

struct get_smape
{
    using ValueType = float;

    __host__ __device__
    double operator()(const ValueType& a, const ValueType& b)
    {
        return fabs((double)b - (double)a) / (double)(fabs(a) + fabs(b));
    }
};

struct bmp_popcount_tuple
{
    using UnsignedIndexType = uint32_t;
    using BMPType = uint64_t;

  __host__ __device__
  float operator()(const thrust::tuple<const UnsignedIndexType &, const BMPType &> x)
  {
      return (float) __builtin_popcountl(x.get<1>()); //TODO not GPU compatible
  }
};

struct bmp_popcount_tuple_noTuple
{
    using UnsignedIndexType = uint32_t;
    using BMPType = uint64_t;

  __host__
  float operator()(const BMPType & x)
  {
      return (float) __builtin_popcountl(x); //TODO not GPU compatible
  }
};

struct sub_mean
{
    float mean_;

    sub_mean(float mean) : mean_(mean) {}

  __host__ __device__
  float operator()(const float &x)
  {
      return (float) (x - mean_) * (x - mean_);
  }
};

template <typename InputType>
float time_spmmBMP_noTuple(const InputType& A_h, const InputType& B_h)
{
    using IndexType = int;
    using ValueType = float;
    using ValueTypeBMP = uint64_t;
    using ElemIndexType = uint32_t;
    using CSRHostBMP = cusp::csr_matrix<IndexType,ValueTypeBMP,cusp::host_memory>;
    using CSRDevBMP = cusp::csr_matrix<IndexType,ValueTypeBMP,cusp::device_memory>;
    using CSRHost = cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>;
    using CSRDev = cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>;
    using COOHostBMP = cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::host_memory>;
    using COODevBMP =  cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::device_memory>;
    using COOHost =    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>;
    using COODev =     cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>;

// ****************** Timings ******************
    unsigned int N = REPETITIONS; //repetitions for timing


    const COOHost A_COO_h(A_h), B_COO_h(B_h);

    uint32_t big_A = thrust::count_if(A_COO_h.values.begin(), A_COO_h.values.end(), greater_equal_absf<float>(HALF_MAX));
    uint32_t big_B = thrust::count_if(B_COO_h.values.begin(), B_COO_h.values.end(), greater_equal_absf<float>(HALF_MAX));

    uint32_t small_A = thrust::count_if(A_COO_h.values.begin(), A_COO_h.values.end(), lesser_equal_absf<float>(HALF_MIN));
    uint32_t small_B = thrust::count_if(B_COO_h.values.begin(), B_COO_h.values.end(), lesser_equal_absf<float>(HALF_MIN));

    // If values are bigger than HALF_MAX then the values become "inf". inf*0=nan, nan*anything=nan. They propagate
    // and break things. If values are smaller than HALF_MIN then the values become 0.
    if (big_A || big_B) {
        printf(
                "BMP method is not possible because A has %d and B has %d elements that are too big to be represented as fp16.\n",
                big_A, big_B);
        return 9998;
    }
    if (small_A || small_B) {
        printf(
                "BMP method is not possible because A has %d and B has %d elements that are too small to be represented as fp16.\n",
                small_A, small_B);
        return 9999;
    }

    COODev A_COO_d(A_COO_h);
    COODev B_COO_d(B_COO_h);

    COODevBMP A_BMP_d;
    COODevBMP B_BMP_d;

    thrust::device_vector<ValueType> A_elems_d;
    thrust::device_vector<ValueType> B_elems_d;
    thrust::device_vector<ValueType> C_elems_d; //This is initialized inside the multiply routine

    thrust::device_vector<ElemIndexType> A_idx_d;
    thrust::device_vector<ElemIndexType> B_idx_d;
    thrust::device_vector<ElemIndexType> C_idx_d; //This is initialized inside the multiply routine

    timer t_conv;
    coo2bmp_noTuple_d(A_COO_d, A_BMP_d, A_elems_d, A_idx_d);
    coo2bmp_noTuple_d(B_COO_d, B_BMP_d, B_elems_d, B_idx_d);
    float time_conversion = t_conv.milliseconds_elapsed();
    printf(" COO to bitmap conversion (for both inputs) time: %lfms\n", time_conversion);

#if DEBUG
    int print_num2 = 100; // How many to print

    std::cout << "A Bmp. ( " << A_BMP_d.num_entries << " )" << std::endl;
    for (int i = 0; i < (A_BMP_d.num_entries < 100 ? A_BMP_d.num_entries : print_num2 ); ++i) {
        std::cout << "[" << A_BMP_d.row_indices[i] << ", " << A_BMP_d.column_indices[i] << "]=["
                << A_idx_d[i] << ", " << A_BMP_d.values[i] << "]  ";
    }
    std::cout << std::endl;

    std::cout << "A Bmp elements.";
    for (int i = 0; i < (A_elems_d.size() < 100 ? A_elems_d.size() : print_num2 ); ++i) {
        std::cout << A_elems_d[i] << " ";
    }
    std::cout << std::endl;
#endif

    COODev test_conv(A_COO_d.num_rows, A_COO_d.num_cols, A_elems_d.size());
    timer t_conv_back;
    bmp2coo_noTuple_d(A_BMP_d, A_elems_d, A_idx_d, test_conv);
    float time_conversion_back = t_conv_back.milliseconds_elapsed();
    printf("Bitmap to COO conversion time x2 (to approximate 2 matrices): %lfms\n", time_conversion_back*2);

#if GPU_WARMUP
    float4* preheat;
    const dim3 BP(8, 8, 4);
    const dim3 GP(30, 30, 30);
    gpuErrchk( cudaMalloc((void**)&preheat, BP.x*BP.y*BP.z*sizeof(float4)) );
    for (int i = 0; i < 20000; ++i) {
        warmup <<< GP, BP >>>(preheat);
    }
#endif

    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        COODevBMP C_BMP_d;
        thrust::device_vector<ValueType> C_elems_d;
        multiplyBmp_noTuple(A_BMP_d, A_elems_d, A_idx_d, B_BMP_d, B_elems_d, B_idx_d, C_BMP_d, C_elems_d, C_idx_d);
    }

    float time_elapsed = t.milliseconds_elapsed() / N;

    printf("Timing of new approach complete\n");

#if GPU_WARMUP
    gpuErrchk(cudaFree(preheat));
#endif


// ****************** Comparison with CUSP ******************

    /* ------------------ Calculate ground truth -------------------*/
    COODev A_ground_d;
    COODev B_ground_d;

    try
    {
        A_ground_d = A_h;
        B_ground_d = B_h;
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

    COODev C_ground_d;
//    cusp::multiply(A_ground_d, B_ground_d, C_ground_d);
    cusp_multiplyGPU(A_ground_d, B_ground_d, C_ground_d);

    COOHost C_ground_h(C_ground_d);

    printf("Calculation of ground truth complete\n");

    /* ------------------ Calculate ours -------------------*/
    COODevBMP C_BMP_d;
    multiplyBmp_noTuple(A_BMP_d, A_elems_d, A_idx_d, B_BMP_d, B_elems_d, B_idx_d, C_BMP_d, C_elems_d, C_idx_d);

    printf("Calculation of new complete\n");

    COOHostBMP C_BMP_h(C_BMP_d);
    thrust::host_vector<ValueType> C_elems_h(C_elems_d);

    COODev C_ours_d(A_COO_d.num_rows, B_COO_d.num_cols, C_elems_d.size());
    bmp2coo_noTuple_d(C_BMP_d, C_elems_d, C_idx_d, C_ours_d);

    COOHost C_ours_h(C_ours_d);

    printf("Conversion of new from BMP to COO complete\n");

    /**/
    COOHost A_ground_h_coo(A_ground_d);
    COOHost B_ground_h_coo(B_ground_d);
    COOHost C_ground_h_coo(C_ground_d);

    if (C_ours_h.num_entries){

    IndexType NNZ_Tiles = thrust::inner_product(
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ground_h.row_indices.begin(), C_ground_h.column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(C_ground_h.row_indices.end(), C_ground_h.column_indices.end()))
                    - 1,
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ground_h.row_indices.begin(), C_ground_h.column_indices.begin())) + 1,
            IndexType(0), thrust::plus<IndexType>(), thrust::not_equal_to<thrust::tuple<IndexType, IndexType> >()) + 1;

    std::cout << "Total count of ground values: " << C_ground_h.num_entries << " Unique (consecutive): "
            << NNZ_Tiles << std::endl;

    NNZ_Tiles = thrust::inner_product(
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ours_h.row_indices.begin(), C_ours_h.column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(C_ours_h.row_indices.end(), C_ours_h.column_indices.end()))
                    - 1,
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ours_h.row_indices.begin(), C_ours_h.column_indices.begin())) + 1,
            IndexType(0), thrust::plus<IndexType>(), thrust::not_equal_to<thrust::tuple<IndexType, IndexType> >()) + 1;

    std::cout << "Total count of new values: " << C_ours_h.num_entries << " Unique (consecutive): "
            << NNZ_Tiles << std::endl;


    //If values at missing or additional slots are bigger than the following threshold, and thus not sufficiently close
    //to 0, they are considered that are additional or missing indeed. As the threshold becomes smaller, more values pass
    ValueType threshold_almost_zero = 0.1;
    //If the absolute difference of the values is smaller than the following threshold they are accepted as matching.
    //As the threshold becomes bigger, more values pass
    ValueType threshold_diff = 0.1;
    //If the ratio of absolute difference of the values to the values is smaller than the following threshold they are
    // accepted as matching. As the threshold becomes bigger, more values pass
    ValueType threshold_diff_ratio = 0.1;

    //Ordering is required for binary searches, and also for comparing the result of RIG with GIR. bmp2coo routine does
    //this.

    /* ----- Search values of Result In Ground truth (RIG) --------*/
    //We take the coordinates of elements from one matrix and we check if there is any value at the same position of
    //the other matrix.
    thrust::host_vector<bool> outputRIG(C_ours_h.num_entries);

    thrust::binary_search(
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ground_h.row_indices.begin(), C_ground_h.column_indices.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ground_h.row_indices.end(), C_ground_h.column_indices.end())),
            thrust::make_zip_iterator(thrust::make_tuple(C_ours_h.row_indices.begin(), C_ours_h.column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(C_ours_h.row_indices.end(), C_ours_h.column_indices.end())),
            outputRIG.begin(), tuple_order());

    //How many slots of the tested have corresponding slots in ground truth
    uint32_t matching_RIG_count = thrust::count(outputRIG.begin(), outputRIG.end(), true);
    //The rest are additional, ie they exist only in the tested
    uint32_t additional_values_count = outputRIG.size() - matching_RIG_count;

    //Save the values of the matching slots in order to compare later, with the result of binary searching the ground
    //truth in the tested
    thrust::host_vector<ValueType> matching_RIG(matching_RIG_count);
    thrust::copy_if(C_ours_h.values.begin(), C_ours_h.values.end(), outputRIG.begin(), matching_RIG.begin(),
            thrust::identity<bool>());

    //Count how many of the additional values exceed the threshold (in order to avoid residues very close to zero or
    //results of cancellation)
    thrust::host_vector<ValueType> additional_values(additional_values_count);
    thrust::remove_copy_if(C_ours_h.values.begin(), C_ours_h.values.end(), outputRIG.begin(), additional_values.begin(),
            thrust::identity<bool>());
    thrust::for_each(additional_values.begin(), additional_values.end(), absolute());
    uint32_t thresh_additional_values_count = thrust::count_if(additional_values.begin(), additional_values.end(),
            thrust::placeholders::_1 > threshold_almost_zero);
    auto thresh_additional_values_max_found = thrust::max_element(additional_values.begin(), additional_values.end());
    ValueType thresh_additional_values_max = 0.f;
    if (additional_values.end() - thresh_additional_values_max_found) {
        thresh_additional_values_max = *thresh_additional_values_max_found;}

    /* ----- Search values of Ground truth In Result (GIR) --------*/
    //We take the coordinates of elements from one matrix and we check if there is any value at the same position of
    //the other matrix.
    thrust::host_vector<bool> outputGIR(C_ground_h.num_entries);

    thrust::binary_search(
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ours_h.row_indices.begin(), C_ours_h.column_indices.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(C_ours_h.row_indices.end(), C_ours_h.column_indices.end())),
            thrust::make_zip_iterator(thrust::make_tuple(C_ground_h.row_indices.begin(), C_ground_h.column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(C_ground_h.row_indices.end(), C_ground_h.column_indices.end())),
            outputGIR.begin(), tuple_order());

    uint32_t matching_GIR_count = thrust::count(outputGIR.begin(), outputGIR.end(), true);
    uint32_t missing_values_count = outputGIR.size() - matching_GIR_count;

    thrust::host_vector<ValueType> matching_GIR(matching_GIR_count);
    thrust::copy_if(C_ground_h.values.begin(), C_ground_h.values.end(), outputGIR.begin(), matching_GIR.begin(),
            thrust::identity<bool>());

    thrust::host_vector<ValueType> missing_values(missing_values_count);
    thrust::remove_copy_if(C_ground_h.values.begin(), C_ground_h.values.end(), outputGIR.begin(), missing_values.begin(),
            thrust::identity<bool>());
    thrust::for_each(missing_values.begin(), missing_values.end(), absolute());
    uint32_t thresh_missing_values_count = thrust::count_if(missing_values.begin(), missing_values.end(),
            thrust::placeholders::_1 > threshold_almost_zero);
    auto thresh_missing_values_max_found = thrust::max_element(missing_values.begin(), missing_values.end());
    ValueType thresh_missing_values_max = 0.f;
    if (missing_values.end() - thresh_missing_values_max_found) {
        thresh_missing_values_max = *thresh_missing_values_max_found;}

    /*------------- Comparison and precision of values -------------*/
    //Based on the assumption that matches of tested in ground truth and matches of ground truth in tested are the same
    //in value and with the same order, we can compare the values of tested and ground truth
    if (matching_RIG_count != matching_GIR_count)
        std::cout<<"ERROR: The results of binary search don't match. A likely reason is many elements with the same indices."
                " Matching after thresholding and precision are unreliable with duplicates."<<std::endl;

    //Count corresponding values that their absolute difference is inside a threshold
    thrust::host_vector<ValueType> absdiff(matching_RIG_count);
    thrust::transform(matching_RIG.begin(), matching_RIG.end(), matching_GIR.begin(), absdiff.begin(), abs_diff<ValueType>());
    uint32_t thresh_matching_values_count = thrust::count_if(absdiff.begin(), absdiff.end(),
            thrust::placeholders::_1 < threshold_diff);

    //Count corresponding values that the ratio of their absolute difference is inside a threshold
    thrust::transform(matching_RIG.begin(), matching_RIG.end(), matching_GIR.begin(), absdiff.begin(), abs_diff_ratio<ValueType>());
    uint32_t thresh_matching_values_count_ratio = thrust::count_if(absdiff.begin(), absdiff.end(),
            thrust::placeholders::_1 < threshold_diff_ratio);
    auto thresh_matching_values_max_found = thrust::max_element(absdiff.begin(), absdiff.end());
    ValueType thresh_matching_values_max = 0.f;
    if (absdiff.end() - thresh_matching_values_max_found) {
        thresh_matching_values_max = *thresh_matching_values_max_found;}

    //Find average precision
    double average_precision = thrust::inner_product(matching_RIG.begin(), matching_RIG.end(), matching_GIR.begin(), 0.,
            thrust::plus<double>(), abs_diffDouble()); //This function is not good for GPUs without double precision
    average_precision /= matching_RIG_count; //TODO maybe use thresholded values for precision

    double average_precision_ratio = thrust::inner_product(matching_RIG.begin(), matching_RIG.end(), matching_GIR.begin(), 0.,
            thrust::plus<double>(), abs_diff_ratioDouble()); //This function is not good for GPUs without double precision
    average_precision_ratio /= matching_RIG_count; //TODO maybe use thresholded values for precision

    double smape = thrust::inner_product(matching_RIG.begin(), matching_RIG.end(), matching_GIR.begin(), 0.,
            thrust::plus<double>(), get_smape()); //This function is not good for GPUs without double precision
    smape /= matching_RIG_count; //TODO maybe use thresholded values for precision

    std::cout << "Matching slots: " << matching_RIG_count << " Additional slots: "
            << additional_values_count << " Missing slots: " << missing_values_count << std::endl;
    std::cout << "After simple thresholding. Matching: " << thresh_matching_values_count << " Additional: "
            << thresh_additional_values_count << " {max: " << thresh_additional_values_max << "}. Missing: "
            << thresh_missing_values_count << " {max: " << thresh_missing_values_max << "}" << std::endl;
    std::cout << "Thresholding based on the ratio of absolute difference to value. Matching: "
            << thresh_matching_values_count_ratio << std::endl;

    std::cout << "Average difference of precision for all matching slots: " << average_precision << std::endl;
    std::cout << "Ratio of average difference of precision for all matching slots: " << average_precision_ratio
            << " {max: " << thresh_matching_values_max << "}" << std::endl;
    std::cout << "Symmetric mean absolute percentage error (SMAPE): " << smape << std::endl;
    //Some times the maximum ratio can be large. It can happen when an element in C is the result of adding opposite
    //numbers. Eg 100090 - 100000 = 90 and 100001 - 100000 = 1. 90 and 1 are very different, although the operands of
    //the subtraction are almost the same.

    /**/
    uint32_t countInf = thrust::count(C_ours_h.values.begin(), C_ours_h.values.end(), ValueType(INFINITY));
    std::cout << "Count of inf in new: " << countInf << std::endl;

    uint32_t countInf_ground = thrust::count(C_ground_h.values.begin(), C_ground_h.values.end(), ValueType(INFINITY));
    std::cout << "Count of inf in ground: " << countInf_ground << std::endl;

#if DEBUG
    uint32_t count0   = thrust::count(C_ours_h.values.begin(), C_ours_h.values.end(), ValueType(0));
    uint32_t countNan = thrust::count(C_ours_h.values.begin(), C_ours_h.values.end(), ValueType(NAN));

    std::cout << "Count of   0 in new (must be 0): " << count0   << std::endl;
    std::cout << "Count of nan in new (must be 0): " << countNan << std::endl;

    uint32_t   count0_ground = thrust::count(C_ground_h.values.begin(), C_ground_h.values.end(), ValueType(0));
    uint32_t countNan_ground = thrust::count(C_ground_h.values.begin(), C_ground_h.values.end(), ValueType(NAN));

    std::cout << "Count of   0 in ground (must be 0): " <<   count0_ground << std::endl;
    std::cout << "Count of nan in ground (must be 0): " << countNan_ground << std::endl;
#endif

        /*Use the various statistics to guestimate if the SpMM was correct*/
        if (countInf == 0 && countInf_ground == 0
                && (missing_values_count == 0
                        || (thresh_missing_values_count < 0.001*matching_RIG_count
                                && thresh_missing_values_max < 1e-8))
                && (additional_values_count == 0
                        || (thresh_additional_values_count < 0.001*matching_RIG_count
                                && thresh_additional_values_max < 0.0003))
                && ((thresh_matching_values_count == matching_RIG_count && average_precision < 0.0001)
                        || (thresh_matching_values_count_ratio == matching_RIG_count && average_precision_ratio < 0.0001
                                && thresh_matching_values_max < 0.1)
                        || ((thresh_matching_values_count == matching_RIG_count && average_precision < 0.0005)
                                && (thresh_matching_values_count_ratio == matching_RIG_count
                                        && average_precision_ratio < 0.0008 && thresh_matching_values_max < 0.1)))) {
            std::cout << "SpMM result successfully tested against ground truth" << std::endl;
        }

    /* Some statistics regarding how many elements per bmp. Median, mean, std.*/
    thrust::host_vector<float> C_bmp_counts(C_BMP_h.num_entries);
    thrust::transform(C_BMP_h.values.begin(), C_BMP_h.values.end(), C_bmp_counts.begin(), bmp_popcount_tuple_noTuple());

    thrust::sort(C_bmp_counts.begin(), C_bmp_counts.end());

    float C_bmp_counts_median = C_bmp_counts[C_bmp_counts.size() / 2];
    float C_bmp_counts_average = C_elems_h.size() / ((float)C_BMP_h.num_entries);

    float C_bmp_counts_std = thrust::transform_reduce(C_bmp_counts.begin(), C_bmp_counts.end(), sub_mean(C_bmp_counts_average),
            0.f, thrust::plus<float>());
    C_bmp_counts_std = sqrtf(C_bmp_counts_std / C_bmp_counts.size());

    std::cout << "Statistics of C BMPs. Median: " << C_bmp_counts_median << " Average: " << C_bmp_counts_average << " std: "
            << C_bmp_counts_std << std::endl;

    /*Bitmap statistics of A*/
    COOHostBMP A_BMP_h(A_BMP_d);
    thrust::host_vector<ValueType> A_elems_h(A_elems_d);

    thrust::host_vector<float> A_bmp_counts(A_BMP_h.num_entries);
    thrust::transform(A_BMP_h.values.begin(), A_BMP_h.values.end(), A_bmp_counts.begin(), bmp_popcount_tuple_noTuple());

    thrust::sort(A_bmp_counts.begin(), A_bmp_counts.end());

    float A_bmp_counts_median = A_bmp_counts[A_bmp_counts.size() / 2];
    float A_bmp_counts_average = A_elems_h.size() / ((float)A_BMP_h.num_entries);

    float A_bmp_counts_std = thrust::transform_reduce(A_bmp_counts.begin(), A_bmp_counts.end(), sub_mean(A_bmp_counts_average),
            0.f, thrust::plus<float>());
    A_bmp_counts_std = sqrtf(A_bmp_counts_std / A_bmp_counts.size());

    std::cout << "Statistics of A BMPs. Median: " << A_bmp_counts_median << " Average: " << A_bmp_counts_average << " std: "
            << A_bmp_counts_std << std::endl;

    } else {
        std::cout << "No test, the result is empty." << std::endl;
    }

    return time_elapsed;
}

template <typename MatrixType, typename InputType>
float time_spmmGPU(const InputType& A,
                const InputType& B)
{
    unsigned int N = REPETITIONS;

    MatrixType A_;
    MatrixType B_;

    try
    {
        A_ = A;
        B_ = B;
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }

#if GPU_WARMUP
    float4* preheat;
    const dim3 BP(8, 8, 4);
    const dim3 GP(30, 30, 30);
    gpuErrchk( cudaMalloc((void**)&preheat, BP.x*BP.y*BP.z*sizeof(float4)) );
    for (int i = 0; i < 20000; ++i) {
        warmup <<< GP, BP >>>(preheat);
    }
#endif

    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        MatrixType C_;
        cusp_multiplyGPU(A_, B_, C_);
    }

    float time_elapsed = t.milliseconds_elapsed();

#if GPU_WARMUP
    gpuErrchk(cudaFree(preheat));
#endif

    return time_elapsed / N;
}


int main(int argc, char ** argv)
{
    typedef int    IndexType;
    typedef float  ValueType;

    typedef cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> CSRHost;
    typedef cusp::csr_matrix<IndexType,ValueType,cusp::device_memory> CSRDev;
    typedef cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> COOHost;
    typedef cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> COODev;

    cudaSetDevice(0);

    //XXX !!!!!! CSR format doesn't work with symmetric matrix marker format !!!!!!
    COOHost A;
    COOHost B;

    if (argc == 1)
    {
        // no input file was specified, generate an example
    	  uint32_t dimension = 1000;
        cusp::gallery::poisson5pt(A, dimension, dimension);
        cusp::gallery::poisson5pt(B, dimension, dimension);
    }
    else if (argc == 2)
    {
        // no input file was specified, generate an example
        cusp::io::read_matrix_market_file(A, argv[1]);
        B = A;
    }
    else if (argc == 3)
    {
        // input files were specified, read them from disk
        //XXX !!!!!! CSR format doesn't work with symmetric matrix marker format !!!!!!
        cusp::io::read_matrix_market_file(A, argv[1]);
        cusp::io::read_matrix_market_file(B, argv[2]);
    }


    std::cout << "Input matrix A has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n";
    std::cout << "             B has shape (" << B.num_rows << "," << B.num_cols << ") and " << B.num_entries << " entries" << "\n\n";

    printf("\n\n");

    printf("Device Sparse Matrix-Matrix Multiply (milliseconds per multiplication)\n");
    printf(" Bmp Device | %9.2f ms\n", time_spmmBMP_noTuple(A, B));

    printf("CUSP Device | %9.2f ms\n", time_spmmGPU<COODev>(A, B));

    return 0;
}

