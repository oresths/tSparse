# bb_segsort (segmented sort): Fast Segmented Sort on GPUs
This repository provides a fast segmented sort on NVIDIA GPUs. The library contains many parallel kernels for different types of segments. In particular, the kernels for solving short/medium segments are automatically generated to efficiently utilize registers in GPUs. More details about the kernels and code generation can be found in our paper (below).

* Contact Email: kaixihou@vt.edu


## Citing Our Work:
* PlainText:  
Fast Segmented Sort on GPUs. 
Kaixi Hou, Weifeng Liu, Hao Wang, Wu-chun Feng.
In Proceedings of the 31th International Conference on Supercomputing (ICS), 
Chicago, USA, 
June 2017.
* Bibtex:  
@InProceedings{Hou:Segsort,  
	author =	{Hou, Kaixi and Liu, Weifeng and Wang, Hao and Feng, Wu-chun},  
	title = 	{Fast Segmented Sort on GPUs},  
	booktitle =	{31th International Conference on Supercomputing (ICS)},  
	address =	{Chicago, USA},  
	month =	{June},  
	year =	{2017},  
}

## Usage:
You can make changes to the Makefile accordingly. Especially, you need to change the ARCH according to your GPU platform. For example, if you are using the P100, you should update ARCH to 61. The main.cu contains an example of how to use it.

The following shows how to run the example codes.
```
$ make
```
After compilation, run the executable as: 
```
$ ./main.out
```

To use the segmented sort (**bb_segsort**), you just need to include the bb_sort.h (with other *.h files).  Note, bb_segsort utilizes an unstable sorting network as the building block; thus, equivalent elements are not guaranteed to keep the original relative order. We plan to provide a version to support stable sort in the future. 

## License: 
Please refer to the included LICENSE file.
