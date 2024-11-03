#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080
#define MAX_CLIENTS 30

int main() {
  int server_fd, client_fd, max_sd, activity;
  int client_sockets[MAX_CLIENTS] = {0};
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);
  char buffer[1025];  // Data buffer of 1K + 1 for null-termination

  // Set of socket descriptors
  fd_set readfds;

  // Create a socket
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  // Set socket options
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&opt,
                 sizeof(opt)) < 0) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }

  // Type of socket created
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT);

  // Bind the socket to the port 8080
  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  // Listen to the socket, with 5 max connection requests queued
  if (listen(server_fd, 5) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  printf("Listening on port %d \n", PORT);

  while (1) {
    // Clear the socket set
    FD_ZERO(&readfds);

    // Add master socket to set
    FD_SET(server_fd, &readfds);
    max_sd = server_fd;

    // Add child sockets to set
    for (int i = 0; i < MAX_CLIENTS; i++) {
      // Socket descriptor
      client_fd = client_sockets[i];

      // If valid socket descriptor then add to read list
      if (client_fd > 0) {
        FD_SET(client_fd, &readfds);
      }

      // Highest file descriptor number, need it for the select function
      if (client_fd > max_sd) {
        max_sd = client_fd;
      }
    }

    // Wait for an activity on one of the sockets, timeout is NULL, so wait
    // indefinitely
    activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);

    if ((activity < 0) && (errno != EINTR)) {
      printf("select error");
    }

    // If something happened on the master socket, then it's an incoming
    // connection
    if (FD_ISSET(server_fd, &readfds)) {
      if ((client_fd = accept(server_fd, (struct sockaddr *)&address,
                              (socklen_t *)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
      }

      printf("New connection, socket fd is %d, ip is: %s, port: %d \n",
             client_fd, inet_ntoa(address.sin_addr), ntohs(address.sin_port));

      // Add new socket to array of sockets
      for (int i = 0; i < MAX_CLIENTS; i++) {
        if (client_sockets[i] == 0) {
          client_sockets[i] = client_fd;
          printf("Adding to list of sockets as %d\n", i);
          break;
        }
      }
    }

    // Else it's some IO operation on some other socket
    for (int i = 0; i < MAX_CLIENTS; i++) {
      client_fd = client_sockets[i];

      if (FD_ISSET(client_fd, &readfds)) {
        // Check if it was for closing, and also read the incoming message
        int bytes_read = read(client_fd, buffer, 1024);
        if (bytes_read == 0) {
          // Somebody disconnected, get his details and print
          getpeername(client_fd, (struct sockaddr *)&address,
                      (socklen_t *)&addrlen);
          printf("Host disconnected, ip %s, port %d \n",
                 inet_ntoa(address.sin_addr), ntohs(address.sin_port));

          // Close the socket and mark as 0 in list for reuse
          close(client_fd);
          client_sockets[i] = 0;
        } else {
          // Echo back the message that came in
          buffer[bytes_read] = '\0';
          send(client_fd, buffer, strlen(buffer), 0);
        }
      }
    }
  }

  return 0;
}
