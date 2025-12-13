#!/usr/bin/env python

from qrcode import QRCode
from qrcode.constants import ERROR_CORRECT_Q
from qrcode.image.styles.moduledrawers.svg import SvgPathSquareDrawer
from qrcode.image.svg import SvgPathFillImage

qr = QRCode(
    version=1,
    error_correction=ERROR_CORRECT_Q,
    box_size=10,
    border=1,
    image_factory=SvgPathFillImage,
)
qr.add_data(
    "https://docs.google.com/document/d/1t8QjnHjA6Dj1atmErGQ_0C71qGGAhu_u4OMdjG1XPm0"
)
img = qr.make_image(module_drawer=SvgPathSquareDrawer())
img.save(stream="qrcode.svg")
