# Zoo-Distance-Detector-YoloV4-Darknet
A custom detection tool to identify maras and anteater in the muenster zoo using YoloV4 on AlexyAB Darknet.

## Downloads
Additional testing fotos and videos including a full weight file can be downloaded here: https://1drv.ms/u/s!Au1UAvkWbNnQzrNgiE4gudw_1g6FBg?e=IABKJe

A precompiled Window x64 Darknet Version for Cuda 11.2 with CDNN and OpenCV 4.5.1 can be downloaded here: https://1drv.ms/u/s!Au1UAvkWbNnQzrMt0JUrSnrfj4i1ZA?e=SWGx0j



## Requirements

Darknet: https://github.com/AlexeyAB/darknet

Python 3.7.7 (Very important; import darknet stoped working with 3.8)

    pip install --upgrade pip
    
    pip install autopep8 , cython , imageio , lxml , matplotlib , numpy , pillow , pypi , pyqt5 , scipy , utils , opencv-python    



# Installation for Darknet in Windows10 in March 2021 (will change in the future) for GTX 1080ti GPU

Follow the darknet AlexyAB Guide maybe with these versions:

  ffmpeg 4.2
  
  Git 2.30.1
  
  CMake 3.2
  
  Ninja 1.10.2
  
  VS2017 C++ CV15 (VS2019 wasn't working for me)
  
  OpenCV 4.5.1
  
  Cuda 11.2 + 10.2 (Install both for compiling the dlls)
  
  +CUDNN
  


## Windows Path and Variables

C:\ffmpeg

C:\ninja

C:\vcpkg

C:\Program Files\Git\cmd

C:\Program Files\CMake\bin


C:\Python37\Scripts\

C:\Python37\



C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR

C:\Program Files\NVIDIA Corporation\Nsight Compute 2020.2.0\

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\libnvvp


C:\OpenCV\build\install\x64\vc15\bin


OPENCV_DIR C:\OpenCV\build\install\


CUDA_PATH C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2

	CUDA_PATH_V11_2 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
	
	CUDA_PATH_V10_2 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
	
	
VCPKG_ROOT C:\vcpkg

VCPKG_DEFAULT_TRIPLET x64-windows


FORCE_CPU False




## Darknet compiling with DLLs

1.  CMake OpenCV + Contrib with opencv_world

	    ALL_BUILD.vcxproj (open VS2017 as admin) Release x64 -> All_Build , Install (won't work with Py39)
	
2.  Add dlls to ;build/darknet/x64 + cudnn64_8.dll , opencv_videoio_ffmpeg451_64.dll , opencv_world451.dll	
	
	  darknet.vcxproj open with editor, change CUDA Version to 11.2
	  
    yolo_cpp_dll.vcxproj open with editor, change CUDA Version to 10.2
    
    Build Release x64: yolo_cpp_dll_no_gpu.vcxproj (rename after building or it will be overwritten), yolo_cpp_dll.vcxproj	
    

3.  Build Release x64: darknet.sln

    Properties: 
    
      C/C++ General / Additional Libs: C:\OpenCV\build\install\include
      
      Preprossesor: -CUDNN_HALF (if not 3000 Series)
      
      CUDA / Device: compute_61,sm_61 (For GTX 1080ti, or look up)
      
      
      
  
# Running Test

Images:

./darknet.exe detector test data/zoo.data cfg/yolov4-tiny-zoo.cfg backup/yolov4-tiny-zoo_4000.weights data/mara_baer1.png

./darknet.exe detector test data/zoo.data cfg/yolo-zoo.cfg backup/yolo-zoo_7000.weights data/mara_baer1.png

./darknet.exe detector test data/zoo.data cfg/yolo-zoo.cfg backup/yolo-zoo_7000.weights data/mara_baer2.png

./darknet.exe detector test data/zoo.data cfg/yolo-zoo.cfg backup/yolo-zoo_7000.weights data/mara_baer3.png

./darknet.exe detector test data/zoo.data cfg/yolo-zoo.cfg backup/yolo-zoo_7000.weights data/mara_baer4.png


Videos:

Video Test:

./darknet.exe detector demo data/zoo.data cfg/yolo-zoo.cfg backup/yolo-zoo_7000.weights Zoo_Test_Video.mp4

./darknet.exe detector demo data/zoo.data cfg/yolov4-tiny-zoo.cfg backup/yolov4-tiny-zoo_4000.weights Zoo_Test_Video.mp4




## Author

[Jan-Patrick Bollow](https://github.com/JBollow)
