# raw_can_write.py
import can
import struct
import time

def open_bus(bustype='socketcan', channel='vcan0', bitrate=500000):
    """Open a python-can Bus. Change bustype/channel for PCAN or others."""
    return can.Bus(interface=bustype, channel=channel, bitrate=bitrate)

def pack_value(value, size_bytes=1, signed=False, endian='little'):
    """Return bytes for an integer value with given size and endianness."""
    fmt = {1:'B', 2:'H', 4:'I', 8:'Q'}[size_bytes]
    if signed:
        fmt = fmt.lower()
    # struct uses native endianness with <> for explicit
    prefix = '<' if endian == 'little' else '>'
    return struct.pack(prefix + fmt, value)

def write_register_by_payload(bus, arb_id, register_addr, value,
                              value_size=1, signed=False, endian='little',
                              is_extended_id=False, expect_response_id=None, timeout=1.0):
    """
    Common vendor pattern: send CAN frame where data = [register_addr, <value bytes>].
    Parameters:
      - arb_id: arbitration ID to send to
      - register_addr: one-byte register identifier inside payload (0-255)
      - value: integer value to write
      - value_size: 1/2/4 (bytes)
      - expect_response_id: if set, wait for that ID as confirmation
    """
    value_bytes = pack_value(value, size_bytes=value_size, signed=signed, endian=endian)
    data = bytes([register_addr]) + value_bytes
    msg = can.Message(arbitration_id=arb_id,
                      is_extended_id=is_extended_id,
                      data=data)
    bus.send(msg)
    print(f"Sent -> ID=0x{arb_id:X} DATA={data.hex()}")
    if expect_response_id:
        resp = bus.recv(timeout)
        if resp and resp.arbitration_id == expect_response_id:
            print("Received response:", resp)
            return resp
        else:
            print("No response or unexpected response")
            return None

def write_register_as_message(bus, register_id, data_bytes, is_extended_id=False):
    """
    Alternative: register_id itself is the CAN arbitration ID and the payload is data_bytes.
    """
    msg = can.Message(arbitration_id=register_id,
                      is_extended_id=is_extended_id,
                      data=data_bytes)
    bus.send(msg)
    print(f"Sent message ID=0x{register_id:X} DATA={data_bytes.hex()}")


# Example usage:
if __name__ == '__main__':
    bus = open_bus(bustype='socketcan', channel='vcan0')
    # Example A: device expects [reg_addr, value]
    write_register_by_payload(bus,
                              arb_id=0x200,           # device listening ID
                              register_addr=0x10,     # your "register" inside payload
                              value=1234,
                              value_size=2,
                              signed=False,
                              endian='little',
                              expect_response_id=0x201)  # optional confirmation ID

    # Example B: register_id is the message ID itself. Send 4 bytes value directly.
    write_register_as_message(bus, register_id=0x300, data_bytes=pack_value(0xDEADBEEF,4))
