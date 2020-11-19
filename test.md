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
   
   윈도우 시스템 파일, 레지스트리, 프로그램, 스크립트, 배치 파일 및 기타 유형의 실행 파일   
   
   문서, 전자 메일, 사진, 음악 파일과 같은 개인 파일은 변경되지 않음   
   
<img width="600" src="https://user-images.githubusercontent.com/42834364/99343630-9cddf580-28d1-11eb-9554-6b99a4d69e55.png"></img>

-  HKLM\SYSTEM\ControlSet001\Control\BackupRestore\FilesNotToSnapshot   
   
   해당 레지스트리 키를 이용하여 섀도 복사본에서 특정파일 제외 가능   
   
   
 -------------------------
 
## 2. 시스템 복원
  
 1) 복원 지점 확인
  
    ![복원지점확인](https://user-images.githubusercontent.com/42834364/99348172-44602580-28dc-11eb-8d84-280a7acbb108.png)
  
    System > System protection > System Restore <br/>
   
    ![14여러복원지점](https://user-images.githubusercontent.com/42834364/99343634-9cddf580-28d1-11eb-963f-0b9e04726582.JPG)
  
    프로그램이 설치되었을 때 자동으로 생성된 복원 지점 확인 가능
 </br>
 
 2) 새 복원 지점 Before sample 추가
 
    ![복원지점생성](https://user-images.githubusercontent.com/42834364/99348171-432ef880-28dc-11eb-81a6-ad97ac87cfd2.png)
</br>

 3) 생성한 Before sample로 시스템 복원
 
    ![시스템복원](https://user-images.githubusercontent.com/42834364/99348173-44f8bc00-28dc-11eb-8383-c8d6055d29eb.png)

    Yes를 누르면 복원이 시작되고 재부팅됨
</br>

 4) 시스템 복원 완료
 
    <img width="600" src="https://user-images.githubusercontent.com/42834364/99343622-9b143200-28d1-11eb-89b7-e18a70237630.png"></img>


---------------------------------------

## 3. VssAdmin

-  VssAdmin : Windows 운영 체제에서 VSS 작업을 위해 제공하는 도구

<img width="346" alt="1_basic-1로 이동" src="https://user-images.githubusercontent.com/42834364/99679785-b04eb380-2abf-11eb-86e5-41e45d47236c.png">

  basic-1 으로 이동
</br></br>

<img width="1680" alt="2_gmm실행" src="https://user-images.githubusercontent.com/42834364/99679790-b17fe080-2abf-11eb-8f63-974e843a1afa.png">
   
  gmm 실행
</br></br>

<img width="1680" alt="3_감연된 파일" src="https://user-images.githubusercontent.com/42834364/99679792-b2b10d80-2abf-11eb-8794-561783051a3b.png">

  감염된 파일
</br></br>

<img width="446" alt="4_cmd실행" src="https://user-images.githubusercontent.com/42834364/99679798-b3e23a80-2abf-11eb-890e-72308db88310.png">

  관리자 권한으로 cmd 실행
</br></br>

<img width="487" alt="5_list출력" src="https://user-images.githubusercontent.com/42834364/99679803-b47ad100-2abf-11eb-92c4-0aca1ad01876.png">   

  vssadmin list shadows : 현재 가지고 있는 볼륨 섀도 복사본 목록 출력
    
  해당 섀도 복사본이 생성된 시점과 파일에 접근할 때 필요한 "Shadow Copy Volume"확인   
</br></br>

<img width="473" alt="6_링크" src="https://user-images.githubusercontent.com/42834364/99679808-b5136780-2abf-11eb-8877-5c030b255fc3.png">

  해당 섀도 복사본에 대한 심볼릭 링크를 생성할 것임
   
  mklink 명령어를 통해 C:\vsc에 링크 걸기
</br></br>

![12링크된VSC확인](https://user-images.githubusercontent.com/42834364/99343627-9c455f00-28d1-11eb-8ec3-b608fd1a238e.png)
<img width="747" alt="7_링크확인" src="https://user-images.githubusercontent.com/42834364/99679812-b5136780-2abf-11eb-852d-f7b7e2fc5bdd.png">
  
  C:\vsc 디렉터리로 이동하면 복사되어 있는 해당 시점의 C:\ 확인 가능
</br></br>

<img width="789" alt="8_기존 파일 확인" src="https://user-images.githubusercontent.com/42834364/99679814-b5abfe00-2abf-11eb-8e13-ff0b6aac0d3b.png">

  암호화 되지 않은 기존 파일 확인 
</br></br>

<img width="1680" alt="9_배경화면에 복붙" src="https://user-images.githubusercontent.com/42834364/99679815-b6449480-2abf-11eb-9496-0347d603ea87.png">

  배경화면으로 다시 이동시켜 복원 완료

  작업 종료 후 "rmdir c:\vsc" 입력하여 심볼릭 링크 제거

