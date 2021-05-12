import pytesseract


def post_process(name):
    """
    Post process output, output contains just alphabetic characters

    Args:
        name: str

    Returns:
        str
    """
    name = name.split("\n")[0]
    res_name = ''
    for i in range(len(name)):
        if name[i].isalpha():
            res_name = res_name + name[i]
    if res_name.isalpha():
        return res_name


def text_recognize(img, tesseract_path):
    """
    Use pytesseract model for text recognition
    Args:
        img: cv2
        tesseract_path: str

    Returns:
        str
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    player_name = pytesseract.image_to_string(img, timeout=2)
    return post_process(player_name)
