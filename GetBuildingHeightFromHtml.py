'''
此代码专用于通过Chrome下载CNBH数据
Date: 2023/05/27
author: Lsq
'''
import os
from selenium import webdriver
from time import sleep
if __name__ == '__main__':
    chrome_driver = 'C:\Program Files\Google\Chrome\Application\chromedriver_win32\chromedriver.exe'
    DATA_PATH = r"C:/Users/Dell/AppData/Local/Google/Chrome/User Data/"
    binary_location = "C:/Program Files/Google\/Chrome/Application/chrome.exe"
    option = webdriver.ChromeOptions()
    option.add_experimental_option("prefs", {
        # "download.default_directory": DOWNLOAD_PATH,  # 默认下载路径
        "profile.default_content_settings.popups": 0,  # 设置为0禁止弹出窗口
        # "profile.managed_default_content_settings.images": 2 #不加载图片的情况下，可以提升速度
    })
    option.add_argument("--window-size=1280,800")  # 窗口大小
    option.add_experimental_option("excludeSwitches", ["enable-automation"])
    option.add_experimental_option('useAutomationExtension', False)
    # 启用带插件的浏览器
    option.add_argument("--user-data-dir=" + DATA_PATH)
    option.binary_location = binary_location
    driver = webdriver.Chrome(executable_path=chrome_driver, options=option)

    login = False
    for i in range(73, 137, 2):
        for j in range(15, 53, 2):
            name = f'CNBH10m_X{i}Y{j}.tif'
            url = f'https://zenodo.org/record/7827315/files/{name}'
            print(f"准备处理 {name} 图片..")
            driver.get(url)
            print('DOWNLOAD: %s' % name)
            if not login:  # 首次访问要有个身份验证的过程时间长一点
                sleep(20)
                login = True
            else:
                sleep(2)
    print('All done !')
    driver.quit()