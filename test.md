# VSS 가이드

## 1. 볼륨 섀도 복사본 서비스

-  Volume Shadow Copy Service (VSS)   
    
   시스템을 이전에 설정된 복원 지점으로 되돌려 복원하는 기능   
   
        
-  복원 지점이 자동으로 생성될 때   
   
   애플리케이션 설치 시 : 설치 이전 지점 생성   
   윈도우 업데이트, 자동 업데이트 설치 시 : 설치 이전 지점 생성    
   시스템 복원 시 : 복원 작업 전 생성   
   사용자가 수동 생성   
   
   
-  복원 시 변경되는 것   
   
   윈도우 시스템 파일, 레지스트리, 프로그램,   
   스크립트, 배치 파일 및 기타 유형의 실행 파일   
   
   문서, 전자 메일, 사진, 음악 파일과 같은 개인 파일은 변경되지 않음   
   
   
![13섀도복사본에서특정파일제외](https://user-images.githubusercontent.com/42834364/99343630-9cddf580-28d1-11eb-9554-6b99a4d69e55.png)

-  HKLM\SYSTEM\ControlSet001\Control\BackupRestore\FilesNotToSnapshot   
   
   해당 레지스트리 키를 이용하여 섀도 복사본에서 특정파일 제외 가능   
   
   
 -------------------------
 
## 2. 시스템 복원
  
  1) 복원 지점 확인
  
  <div>
  <img width="600" src="https://user-images.githubusercontent.com/42834364/99343605-964f7e00-28d1-11eb-8bf1-336eb174dc79.png">
  <img width="400" src="https://user-images.githubusercontent.com/42834364/99343615-98b1d800-28d1-11eb-9e09-cc8280981828.png">
  </div>
<br/>
   System > System protection > System Restore
<br/>
   
  ![14여러복원지점](https://user-images.githubusercontent.com/42834364/99343634-9cddf580-28d1-11eb-963f-0b9e04726582.JPG)
  
   프로그램이 설치되었을 때 자동으로 생성된 복원 지점 확인 가능
 
 
 2) 새 복원 지점 Before sample 추가
 
 ![2복원지점생성버튼](https://user-images.githubusercontent.com/42834364/99343609-9780ab00-28d1-11eb-91dc-37c42bbc7ba2.png)
 
 ![3BeforeSample생성](https://user-images.githubusercontent.com/42834364/99343612-98194180-28d1-11eb-8ed6-9a0aada23d36.png)


 3) 생성한 Before sample로 시스템 복원
 
 ![5시스템복원클릭](https://user-images.githubusercontent.com/42834364/99343615-98b1d800-28d1-11eb-9e09-cc8280981828.png)
 
 ![4비포샘플복원지점선택](https://user-images.githubusercontent.com/42834364/99343614-98194180-28d1-11eb-9247-18858e343adc.png)
 
 ![6복원예스누르기](https://user-images.githubusercontent.com/42834364/99343616-994a6e80-28d1-11eb-8b87-bf5fca7b5728.png)
 
   Yes를 누르면 복원이 시작되고 재부팅됨

 ![8복원중](https://user-images.githubusercontent.com/42834364/99343619-99e30500-28d1-11eb-8bf7-56cb86bcfda4.png)


 4) 시스템 복원 완료
 
 ![9복원완료](https://user-images.githubusercontent.com/42834364/99343622-9b143200-28d1-11eb-89b7-e18a70237630.png)


---------------------------------------

## 3. VssAdmin

-  VssAdmin : Windows 운영 체제에서 VSS 작업을 위해 제공하는 도구

 ![10vssadmin_list_shadows](https://user-images.githubusercontent.com/42834364/99343625-9bacc880-28d1-11eb-9898-50608aec966a.png)

   관리자 권한으로 cmd 실행
   vssadmin list shadows : 현재 가지고 있는 볼륨 섀도 복사본 목록 출력
  
 ![11심볼릭링크걸기](https://user-images.githubusercontent.com/42834364/99343626-9bacc880-28d1-11eb-8d10-156cd0752d27.png)
 
   해당 섀도 복사본이 생성된 시점과 파일에 접근할 때 필요한 "Shadow Copy Volume"확인   

   해당 섀도 복사본에 대한 심볼릭 링크를 생성할 것임
   mklink 명령어를 통해 C:\vsc에 링크 걸기
  
 ![12링크된VSC확인](https://user-images.githubusercontent.com/42834364/99343627-9c455f00-28d1-11eb-8ec3-b608fd1a238e.png)
  
   C:\vsc 디렉터리로 이동하면 복사되어 있는 해당 시점의 C:\ 확인 가능

   작업 종료 후 "rmdir c:\vsc" 입력하여 심볼릭 링크 제거

