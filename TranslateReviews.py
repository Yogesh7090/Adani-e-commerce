#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:29:19 2024

@author: samridhishukla
"""

from langdetect import detect
from googletrans import Translator
def translate_to_english(text):
    try:
        detected_language = detect(text)

        if detected_language != 'en':
            translator = Translator()
            translation = translator.translate(text, dest='en')
            return translation.text
        else:
            return text  
    except:
        return text  
 
 
df = pd.read_excel(input_excel_file)
 
 
df['content'] = df['content'].apply(translate_to_english)
 
 
df['replyContent'] = df['replyContent'].apply(translate_to_english)