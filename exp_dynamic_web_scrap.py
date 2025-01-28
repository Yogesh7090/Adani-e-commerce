from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# URL of the website
url = 'https://www.ambitionbox.com/reviews/adani-group-reviews'

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the URL
driver.get(url)

# Find all review elements
reviews = driver.find_elements(By.CLASS_NAME, "review-content-cont")

# Print the text content of each review
for i, review in enumerate(reviews, start=1):
    print(f"Review {i}: {review.text}")

# Close the WebDriver
driver.quit()
