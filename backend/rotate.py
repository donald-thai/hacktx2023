def caesar_cipher(text, shift):
    encrypted_text = ""
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            char = char.lower()
            shifted_char = chr(((ord(char) - ord('a') + shift) % 26) + ord('a'))
            if is_upper:
                shifted_char = shifted_char.upper()
            encrypted_text += shifted_char
        else:
            encrypted_text += char
    return encrypted_text

def rotateFile(file, shift):
    with open(file, 'r') as f:
        text = f.read()
    encrypted_text = caesar_cipher(text, shift)
    with open(file, 'w') as f:
        f.write(encrypted_text)

def removeNonAscii(text):
    new_text =  ''.join([i if ord(i) < 128 else '' for i in text])
    with open("backend/data.txt", "w") as f:
        f.write(new_text)

def replaceCharWithSpecialCharacter(text, targetChar, specialChar):
    new_text = ""
    for char in text:
        if char == targetChar:
            new_text += specialChar
        else:
            new_text += char
    with open("backend/data.txt", "w") as f:
        f.write(new_text)