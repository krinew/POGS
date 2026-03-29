import dynamixel_sdk as dxl

port = "/dev/ttyUSB0"
port_handler = dxl.PortHandler(port)
packet_handler = dxl.PacketHandler(2.0)

if not port_handler.openPort():
    print(f"Failed to open port {port}")
    exit(1)

baudrates = [57600, 115200, 1000000, 2000000, 3000000, 4000000]

for baud in baudrates:
    if not port_handler.setBaudRate(baud):
        print(f"Failed to set baudrate {baud}")
        continue
    
    print(f"Probing baudrate: {baud}")
    dst, comm = packet_handler.broadcastPing(port_handler)
    if comm == dxl.COMM_SUCCESS:
        print(f"  Found motors on baud {baud}:")
        for id_, data in dst.items():
            model = data[0] if isinstance(data, list) and len(data) > 0 else data
            print(f"    ID: {id_}, Model: {model}")
    else:
        print(f"  Ping failed: {packet_handler.getTxRxResult(comm)}")

port_handler.closePort()
