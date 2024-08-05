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

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_addr.S_un.S_addr = INADDR_ANY;
	addr.sin_port = htons(9999);

	bind(s, (sockaddr*)&addr, sizeof(addr));
	//tcp:send udp:sendto
	char buf[512];
	sockaddr_in addrClient;
	int sockLen = sizeof(sockaddr_in);
	cout << "UDP SERVER IS START AND BIND TO 9999" << endl;
	while (true) {
		int recvLen = recvfrom(s, buf, 512, 0, (sockaddr*)&addrClient, &sockLen);
		if (recvLen > 0) {
			cout << buf << endl;
		}


	}
	closesocket(s);
	WSACleanup();

	return 0;
}