from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import re
import os
import json
from urllib.parse import urlparse
import csv
import html

url = "https://law.cityofsanmateo.org/us/ca/cities/san-mateo/code/27"
parsed_url = urlparse(url)
base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
print(base_url)

def transform_structure(data):
    output = []
    chapter_index = 1

    for chapter_title, sections in data.items():
        # Extract readable title
        match = re.match(r"Chapter\s+([\d.]+)\s+(.*)", chapter_title)
        # chapter_code = match.group(1) if match else f"{chapter_index:02d}"
        readable_title = match.group(2).title() if match else chapter_title

        article = {
            "title": f"Chapter {str(chapter_index)} {readable_title}",
            "dir": f"chapter_{chapter_index}",
            "file": "",
            "sections": []
        }

        for i, section in enumerate(sections, 1):
            section_entry = {
                "title": f"§ {section['heading']}",
                "file": f"{chapter_index}.{i}",
                "section": []
            }
            article["sections"].append(section_entry)

        output.append(article)
        chapter_index += 1

    return output
 
# Set up your WebDriver (example: Chrome)
driver = webdriver.Chrome()  # Make sure chromedriver is in your PATH
driver.get("https://law.cityofsanmateo.org/us/ca/cities/san-mateo/code/27")  # Replace with the actual page URL
 
chapter_links_xpath = "//div[@class='toc-menu']/ul/li/ul/li[27]/ul/li"
 
chapter_links = driver.find_elements(By.XPATH, chapter_links_xpath)
results = {}
 
for iter, chapter_link in enumerate(chapter_links):
    title_count = iter + 1
    chapter_link.click()

    chapter_title_1 = chapter_link.text.strip()
    extracting_number = chapter_title_1.split()
    chapter_number = extracting_number[1]
    time.sleep(3)
 
    chapter_title_xpath = f"//h2[@id='/us/ca/cities/san-mateo/code/{chapter_number}']"
    chapter_title_element = driver.find_elements(By.XPATH, chapter_title_xpath)
 
    if chapter_title_element:
        chapter_title = chapter_title_element[0].text
 
    # Find all <h3> elements matching the zoning code ID pattern
    h3_elements = driver.find_elements(By.XPATH, f"//h3[starts-with(@id, '/us/ca/cities/san-mateo/code/{chapter_number}')]")
 
    
 
    for h3 in h3_elements:
        id = h3.get_attribute("id")
        section_url = base_url+id
        # file_number = f"{chapter_index}.{i}"
        section = {
            "heading": h3.text.strip(),
            "id": h3.get_attribute("id"),
            "url": section_url,
            "html": "" ,
            "text": ""
        }
        paragraphs_html = []
        paragraphs_text = []
        # Start walking through the siblings
        sibling = h3.find_element(By.XPATH, "following-sibling::*[1]")
        
        while sibling.tag_name not in ['h2', 'h3']:
            if sibling.tag_name == 'p':
                # outer_html = sibling.get_attribute('outerHTML')
                paragraphs_text.append(sibling.text.strip())

                paragraphs_html.append(sibling.get_attribute('outerHTML'))
            try:
                sibling = sibling.find_element(By.XPATH, "following-sibling::*[1]")
            except:
                break
 
        section["html"] = " ".join(paragraphs_html)
        section["text"] = " ".join(paragraphs_text)
        ###########################################################
        
        # safe_id = html.escape(, quote=True)


        ################################################################

        section["html"]= f"<div>{section['html']}</div>"
        
        # print("Section HTML",section_html_1)
        # results[chapter_title] = section
        if chapter_title not in results:
            # chapter_title = chapter_title.lower()
            results[chapter_title] = []
        results[chapter_title].append(section)
        # print("results type",results)
 

# for chapter_title, sections in results.items():
#     print(f"Chapter: {chapter_title}")
#     for sec in sections:
#         print(f" - {sec['heading']}")
#         print(f"{sec['html']}\n")


transform_structure_content = transform_structure(results)

with open("chapters_11.json", "w", encoding="utf-8") as f:
    json.dump(transform_structure_content, f, ensure_ascii=False, indent=4)


def extract_chapter_name(title):
    # This regex removes "Chapter <number>" (with optional dots or spaces) at the beginning
    return re.sub(r'^Chapter\s+\d+\.?\s*', '', title, flags=re.IGNORECASE).strip()

def normalize(text):
    return text.strip().rstrip('.').lower()


def chapters_wise_jsons(chapters,results):
    base_dir = "chapter_wise_jsons_11"
    os.makedirs(base_dir, exist_ok=True)

    # === Create files ===
    for i,chapter in enumerate(chapters):
        chapter_title = chapter["title"]
        chapter_title = extract_chapter_name(chapter_title)
        chapter_title = chapter_title.upper()
        chapter_title = normalize(chapter_title)

        chapter_title = next(
                    (key for key in results if normalize(key).endswith(chapter_title)),
                    None
                )

        chapter_dir = os.path.join(base_dir, chapter["dir"])
        os.makedirs(chapter_dir, exist_ok=True)

        if chapter_title not in results:
            print(f"Warning: No data for {chapter_title}")
            continue

        chapter_data = results[chapter_title]

    
        for section in chapter["sections"]:
            heading_clean = section["title"].replace("§", "").strip()
            # heading_clean = extract_chapter_name(heading_clean)
            # heading_clean = heading_clean.upper()

            # Match the heading from results
            match = next((item for item in chapter_data if heading_clean.endswith(item["heading"])), None)
            if not match:
                print(f"Warning: No match for section '{section['title']}'")
                continue

            section_data = {
                "source": "san-mateo",
                "title": section["title"],
                "url": match["url"],
                "html": f"{match['html']}"
            }

            filename = section["file"] + ".json"
            file_path = os.path.join(chapter_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(section_data, f, indent=2, ensure_ascii=False)
    
    print(" Section files created successfully.")





def generate_csv(chapters, results, output_file="san_mateo_sections_22.csv"):
    rows = []
    
    for chapter_index, chapter in enumerate(chapters, 1):
        chapter_title = extract_chapter_name(chapter["title"])
        chapter_title_normalized = normalize(chapter_title)

        # Try to match the chapter title to keys in results
        matched_chapter_key = next(
            (key for key in results if normalize(key).endswith(chapter_title_normalized)),
            None
        )

        if not matched_chapter_key or matched_chapter_key not in results:
            print(f"Warning: No data for chapter '{chapter_title}'")
            continue

        chapter_data = results[matched_chapter_key]
        
        for section_index, section in enumerate(chapter["sections"], 1):
            heading_clean = section["title"].replace("§", "").strip()
            section_url = section.get("url", "")  # assume URL is stored here

            match = next((item for item in chapter_data if heading_clean.endswith(item["heading"])), None)
            if not match:
                print(f"Warning: No match for section '{section['title']}' in CSV export")
                continue

            # Split the text into separate paragraphs if needed
            paragraphs = match["text"].split("\n") if match["text"] else [""]
            for para in paragraphs:
                if para.strip():
                    subsection_number = f"{chapter_index}.{section_index}"
                    zoneomics_url = f"https://zoneomics.com/code/san_mateo/chapter_{chapter_index}#{subsection_number}"

                    row = {
                        "chapter_no": chapter_index,
                        "heading_1": chapter_title,
                        "heading_2": match["heading"],
                        "subsection_number": f"sub-sec-{subsection_number}",
                        "source_url":match["url"],
                        "zoneomics_url": zoneomics_url,
                        "text": para.strip()
                    }
                    rows.append(row)

    # Write to CSV
    with open(output_file, "w", encoding="utf-8", newline='') as csvfile:
        fieldnames = [
            "chapter_no", "heading_1", "heading_2", 
            "subsection_number", "source_url", "zoneomics_url", "text"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV file '{output_file}' created successfully.")



chapters_wise_jsons(transform_structure_content,results)
generate_csv(transform_structure_content, results)
driver.quit()
 