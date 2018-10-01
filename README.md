# Blurtect

## 1. Introduction
  오늘날 SNS (Youtube, Instagram 등)의 사용량이 급증함에 따라, 개인의 일상의 매우 많은 부분을 공유하게 되었습니다. 이로 인해 타인의 모습까지 함께 촬영하고 게시하여 타인의 초상권을 침해하는 문제가 늘어났습니다. 
  
  Blurtect는 이를 해결하기 위해, 딥러닝을 통해 사람들의 얼굴이 노출된 경우 이를 자동으로 인식하고 블러 처리합니다.
  
  Blurtect는 간단한 작업을 통해 영상에서 타인의 초상권을 보호할 수 있습니다.  
## 2. Requirements
  Blurtect는 Ubuntu 16.04 환경에서 python 2.7로 작성되었습니다.
  
  아래 요소들을 이용 전 설치해 주시기 바랍니다.
  
### [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
* [MIT License](https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/LICENSE)
* [installation](https://github.com/DuinoDu/mtcnn)
  
### [OpenFace](https://cmusatyalab.github.io/openface/)
* [Apache 2.0 License](https://github.com/cmusatyalab/openface/blob/master/LICENSE)
* [installation](https://www.popit.kr/openface-exo-member-face-recognition/)

### [OpenCV](https://opencv.org)
* [3-clause BSD License](https://opencv.org/license.html)
* [installation](https://blog.csdn.net/duinodu/article/details/51804642)
  
### [Caffe](https://github.com/BVLC/caffe)
* [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE)
* [installation](https://gist.github.com/nikitametha/c54e1abecff7ab53896270509da80215)


>dilb   
>string  
>numpy  
>pil  
>ffmpy  
>os  
>shutil  
>random

## 3. How to use
  **설치한 caffe/python폴더 안에서 아래 명령어를 실행하거나 다운로드한 zip파일을 압축해제 해주세요.**
  
  `~/caffe/python$ git clone https://github.com/jeenyeo/ssdam_blurtect.git`
  
  **_init_path.py 파일에서 caffe 경로를 자신의 caffe 설치 폴더 경로로 변경해주세요.**
  
  `caffe_path = '/home/jieun/caffe'`
  
  **Blurtect폴더 안에 블러 하지 않을 대상의 짧은 영상의 이름을 'sample_video.mp4'로 저장해 주세요.**
  
  **Blurtect폴더 안에 블러를 적용할 영상의 이름을 'test_video.mp4'로 저장해 주세요.**
  
  **ssdam_blurtect_demo.py 파일을 실행해주세요. 폴더 안에 완성본인 'output_video_.mp4'파일이 생깁니다.**
  
  
## 4. Lisence
  ```
MIT License

Copyright (c) 2018 jeenyeo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
  ```
