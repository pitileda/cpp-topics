import socket
import select
import signal

HOST = "127.0.0.1"
PORT = 8080

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print(f"Listening for connections on {HOST}:{PORT}")

socket_list = [server_socket]
clients = {}  # clients address info
output_buffer = {}  # data waiting to be sent to clients

running = True


def sig_handler(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

try:
    while running:
        read_sockets, write_sockets, exceptions_sockests = select.select(
            socket_list,
            [sock for sock in socket_list if output_buffer.get(sock)],
            socket_list,
            1.0,
        )

        for notified_socket in read_sockets:
            if notified_socket == server_socket:
                client_socket, client_addr = server_socket.accept()
                socket_list.append(client_socket)
                clients[client_socket] = client_addr
                output_buffer[client_socket] = []
                print(f"Accepted new connection from {client_addr[0]}:{client_addr[1]}")
            else:
                message = notified_socket.recv(1024)
                if message:
                    print(
                        f"Recieved from {clients[notified_socket]}: {message.decode('utf-8')}"
                    )
                    output_buffer[notified_socket].append(message)
                else:
                    # close connection
                    socket_list.remove(notified_socket)
                    notified_socket.close()
                    del clients[notified_socket]
                    del output_buffer[notified_socket]

        for notified_socket in write_sockets:
            if output_buffer[notified_socket]:
                print(f"Sending the data to notified socket: {notified_socket}")
                message = output_buffer[notified_socket].pop(0)
                notified_socket.send(message)
finally:
    print(f"Close all sockets")
    for s in socket_list:
        s.close()
    server_socket.close()
