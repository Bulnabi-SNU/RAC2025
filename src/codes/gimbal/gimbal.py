import serial

# out of class
def crc_xmodem(data: bytes) -> bytes:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc.to_bytes(2, 'little')

def to_twos_complement(number: int) -> bytes:
    if number < 0:
        number &= 0xFFFF
    return number.to_bytes(2, 'little')

def format_bytearray(byte_array: bytearray) -> str:
    return ' '.join(f'{byte:02x}' for byte in byte_array)

def is_jetson():
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        return False



# in class
def gimbal_reboot(ser):
    if is_jetson():
        data_fix = bytes([0x55, 0x66, 0x01, 0x02, 0x00, 0x00, 0x00, 0x80, 0x00, 0x01])
        data_crc = crc_xmodem(data_fix)
        packet = bytearray(data_fix + data_crc)
        ser.write(packet)

def gimbal_control_callback(ser,gimbal_pitch):
    if is_jetson():
        data_fix = bytes([0x55, 0x66, 0x01, 0x04, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00])
        data_var = to_twos_complement(10 * int(gimbal_pitch))
        data_crc = crc_xmodem(data_fix + data_var)
        packet = bytearray(data_fix + data_var + data_crc)
        ser.write(packet)

def is_jetson():
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        return False
    




"""
Main code
"""
def main(args = None):
    # do something
    gimbal_pitch = 0.0
    gimbal_counter = 0

    if is_jetson():
        ser = serial.Serial('/dev/ttyGimbal', 115200)
    else:
        ser = serial.Serial('/dev/ttyGimbal', 115200)

    while(True):
        gimbal_control_callback(ser,gimbal_pitch)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)