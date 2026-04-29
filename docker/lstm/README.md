# 사용법

#### 가장 짧은 형태
cd docker/lstm
docker compose up -d --build

#### 또는 프로젝트 루트에서
docker compose -f docker/lstm/docker-compose.yml up -d --build

-d 가 detached 모드라 SSH 세션이 끊겨도 컨테이너는 계속 돌고, restart: "no" 라 학습이 끝나면 끗이 종료됨.

모니터링 / 정리
#### 로그 따라가기 (Ctrl+C 로 져도 컨테이너는 계속 돎)
docker compose -f docker/lstm/docker-compose.yml logs -f   
#### 상태 확인 (running / xited)  
docker compose -f docker/lstm/docker-compose.yml ps   
#### 종료된 컨테이너 정리        
docker compose -f docker/lstm/docker-compose.yml down