from urllib.request import urlretrieve

url = (
    "https://raw.githubusercontent.com/MSUcourses/"
    "Data-Analysis-with-Python/main/Deep%20Learning/Files/quora.txt"
)

print("Скачивание...")
urlretrieve(url, "quora.txt")
print("Готово! Файл сохранён: quora.txt")