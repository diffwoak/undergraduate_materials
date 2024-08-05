#include<WinSock2.h>
#include<Windows.h>
#include<iostream>
#pragma comment(lib,"ws2_32.lib")
using namespace std;
int main() {
	WSADATA d;
	WORD w = MAKEWORD(2, 0);
	WSAStartup(w, &d);
	SOCKET s = socket(AF_INET, SOCK_DGRAM, 0);
	sockaddr_in serverAddr;
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	serverAddr.sin_port = htons(9999);
	char buf[] = "Hello,UDP server";
	sendto(s, buf, strlen(buf), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));
	closesocket(s);
	WSACleanup();
	return 0;
}