from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# URL of the website
url = 'https://www.ambitionbox.com/reviews/adani-group-reviews'

# Initialize the WebDriver
driver = webdriver.Chrome()

# Open the URL
driver.get(url)

# Find all review blocks
review_blocks = driver.find_elements(By.CLASS_NAME, "review-content-cont")  # Update this based on the actual class name

# Initialize lists to store data
topics = []
reviews = []

# Loop through each review block to extract topic and review
for block in review_blocks:
    try:
        # Extract the topic (e.g., the review title or tag)
        topic_element = block.find_elements(By.TAG_NAME, ".Job Security (81)")  # Update this to the actual class name
        topic = topic_element.text

        # Extract the review content
        # review_element = block.find_element(By.TAG_NAME, "review-content-cont")  # Update class name as needed
        # review = review_element.text

        # Append data to lists
        topics.append(topic)
        # reviews.append(review)
    except Exception as e:
        print(f"Error extracting data: {e}")

# Close the WebDriver
driver.quit()

# Create a DataFrame to store the data
df = pd.DataFrame({'Topic': topics})

# Save to a CSV file or display
df.to_csv('adani_group_reviews.csv', index=False)
print(df)
