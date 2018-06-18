#!/usr/bin/env python3

from google_drive_downloader import GoogleDriveDownloader as gdd


MOBILENET_SSD_MODEL = 'https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc'

gdd.download_file_from_google_drive(file_id='0B3gersZ2cHIxRm5PMWRoTkdHdHc',
        dest_path='./MobileNetSSD_deploy.caffemodel',
        unzip=False)

