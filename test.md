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
 
 2) 새 복원 지점 추가
 
    ![복원지점생성 copy](https://user-images.githubusercontent.com/42834364/99688018-aa110500-2ac8-11eb-81ab-75562beac3be.png)
</br>

 3) 생성한 복원 지점으로 시스템 복원
 
    ![시스템복원 copy](https://user-images.githubusercontent.com/42834364/99687675-45ee4100-2ac8-11eb-826d-d4cda54dc9a9.png)

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

<img width="651" alt="gmm실행1" src="https://user-images.githubusercontent.com/42834364/99685798-4ede1300-2ac6-11eb-88ba-51812abed80c.png">
   
  GMM1.exe 실행
</br></br>

<img width="669" alt="감염된파일" src="https://user-images.githubusercontent.com/42834364/99685797-4e457c80-2ac6-11eb-83e2-d69f596f68d1.png">

  GMM1.exe가 감염시킨 파일 확인
</br></br>

<img width="349" alt="cmd실행" src="https://user-images.githubusercontent.com/42834364/99685796-4dace600-2ac6-11eb-84f3-de5234f60485.png">

  관리자 권한으로 cmd 실행
</br></br>

<img width="600" alt="list출력" src="https://user-images.githubusercontent.com/42834364/99685794-4dace600-2ac6-11eb-904e-d615e7eef301.png">   

  vssadmin list shadows : 현재 가지고 있는 볼륨 섀도 복사본 목록 출력
    
  해당 섀도 복사본이 생성된 시점과 파일에 접근할 때 필요한 "Shadow Copy Volume"확인   
</br>

<img width="600" alt="링크걸기" src="https://user-images.githubusercontent.com/42834364/99685789-4d144f80-2ac6-11eb-8d91-b804962eef63.png">

  해당 섀도 복사본에 대한 심볼릭 링크를 생성할 것임
   
  mklink 명령어를 통해 C:\vsc에 링크 걸기
</br></br>

<img width="669" alt="링크확인" src="https://user-images.githubusercontent.com/42834364/99685785-4d144f80-2ac6-11eb-8f94-dffb31a2e91d.png">

  C:\vsc에 링크 걸린 것 확인
</br>

![12링크된VSC확인](https://user-images.githubusercontent.com/42834364/99343627-9c455f00-28d1-11eb-8ec3-b608fd1a238e.png)
  
  C:\vsc 디렉터리로 이동하면 복사되어 있는 해당 시점의 C:\ 확인 가능
</br></br>

<img width="669" alt="기존파일확인" src="https://user-images.githubusercontent.com/42834364/99685782-4c7bb900-2ac6-11eb-8e2a-bcf2dd1a9525.png">

  암호화 되지 않은 상태인 기존 파일 확인 
</br></br>

<img width="755" alt="배경화면복붙" src="https://user-images.githubusercontent.com/42834364/99685773-4ab1f580-2ac6-11eb-8a4e-742b7a5f76eb.png">

  배경화면으로 다시 복사해서 복원 완료

  작업 종료 후 "rmdir c:\vsc" 입력하여 심볼릭 링크 제거

