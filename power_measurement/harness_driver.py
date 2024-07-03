import os
import time
from pynq import Overlay, allocate
from pynq.lib import AxiGPIO
from pynq.ps import Clocks

class MeasurementHarnessOverlay(Overlay):
    def __init__(
            self,
            bitfile_name,
            fclk_mhz=200.0,
            device=None,
            download=True
        ):
        super().__init__(bitfile_name, download=download, device=device)
        self.fclk_mhz = fclk_mhz

        Clocks.fclk0_mhz = self.fclk_mhz

        gpio_ip = self.ip_dict['axi_gpio_0']
        self.gpio = AxiGPIO(gpio_ip)
        self.gpio.setdirection(AxiGPIO.InOut, channel=1)
        self.gpio1 = AxiGPIO(gpio_ip).channel1
        
        # 0: source start (pulse)
        # 1: source stop (pulse)
        # 2: sink enable

        # 3: sink valid
        # 4: sink checksum

    def start_freerunning(self):
        self.gpio1[2].on()

        self.gpio1[0].on()
        self.gpio1[0].off()

    def stop_freerunning(self):
        self.gpio1[1].on()
        self.gpio1[1].off()

        self.gpio1[2].off()

def run_idle(*args, **kwargs):
    frequency = kwargs["frequency"]
    bitstream_path = kwargs["bitstream_path"]

    print("Loading overlay/bitstream")
    ol = MeasurementHarnessOverlay(bitstream_path, fclk_mhz=frequency)
    print("Loaded overlay/bitstream")

    print("Waiting idle for 30s")
    time.sleep(30)
    print("Stopped test")

def run_free(*args, **kwargs):
    frequency = kwargs["frequency"]
    bitstream_path = kwargs["bitstream_path"]

    print("Loading overlay/bitstream")
    ol = MeasurementHarnessOverlay(bitstream_path, fclk_mhz=frequency)
    print("Loaded overlay/bitstream")

    ol.start_freerunning()
    print("Started freerunning for 30s")
    time.sleep(30)
    ol.stop_freerunning()
    print("Stopped freerunning")

if __name__ == "__main__":
    print("Loading overlay/bitstream")
    ol = MeasurementHarnessOverlay("experiments/23_08_22-22_57-mvau_harness/run_0.bit")
    print("Loaded overlay/bitstream")

    ol.start_freerunning()
    print("Started test")
    time.sleep(30)
    ol.stop_freerunning()
    print("Stopped test")
