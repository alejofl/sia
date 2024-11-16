import numpy as np


class Utils:
    @staticmethod
    def parseFont(font):
        letters = []
        for f in font:
            bits = []
            for row in f:
                bits.append((row & 0b00010000) >> 4)
                bits.append((row & 0b00001000) >> 3)
                bits.append((row & 0b00000100) >> 2)
                bits.append((row & 0b00000010) >> 1)
                bits.append(row & 0b00000001)
            letters.append(np.array(bits))
        return letters
