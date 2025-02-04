# CODEALPHA
# Task#1: Language Translation Tool
# Objective: Develop a simple language translation tool that translates text from one language to another. Use machine translation techniques and pre-trained models like Google Translate API or Microsoft Translator API to translate text.

# This is a GUI Based Application. Built it with a GUI.
# ******************************* LANGUAGE TRANSLATION TOOL ********************************************************

import tkinter as tk
from tkinter import ttk
from googletrans import Translator, LANGUAGES

class LanguageTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Translator")

        self.translator = Translator()

        # Create input text box
        self.input_text = tk.Text(root, height=10, width=50)
        self.input_text.pack(pady=10)

        # Create a dropdown for selecting source language
        self.source_lang = ttk.Combobox(root, values=list(LANGUAGES.values()))
        self.source_lang.set("Select Source Language")
        self.source_lang.pack(pady=5)

        # Create a dropdown for selecting target language
        self.target_lang = ttk.Combobox(root, values=list(LANGUAGES.values()))
        self.target_lang.set("Select Target Language")
        self.target_lang.pack(pady=5)

        # Create a button for translation
        self.translate_button = tk.Button(root, text="Translate", command=self.translate_text)
        self.translate_button.pack(pady=10)

        # Create output text box
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.pack(pady=10)

    def translate_text(self):
        source_lang = self.source_lang.get()
        target_lang = self.target_lang.get()
        text_to_translate = self.input_text.get("1.0", tk.END)

        # Find language codes
        source_lang_code = [key for key, value in LANGUAGES.items() if value == source_lang][0]
        target_lang_code = [key for key, value in LANGUAGES.items() if value == target_lang][0]

        # Translate text
        translation = self.translator.translate(text_to_translate, src=source_lang_code, dest=target_lang_code)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, translation.text)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageTranslatorApp(root)
    root.mainloop()
