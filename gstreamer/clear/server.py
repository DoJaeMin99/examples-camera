import socket

# UDP 서버의 IP 주소와 포트
server_ip = '192.168.0.188'
server_port = 22

# UDP 서버 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 소켓과 IP 주소, 포트를 바인딩
server_socket.bind((server_ip, server_port))

print("Start UDP server.")

while True:
    # 데이터 수신
    data, client_address = server_socket.recvfrom(1024)
    print("Data from client:", data.decode())

    # 응답 데이터 생성
    # response = "Message received!"

    # # 응답 데이터를 클라이언트로 전송
    # server_socket.sendto(response.encode(), client_address)