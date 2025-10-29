from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain.text_splitter import RecursiveCharacterTextSplitter
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (ElementClickInterceptedException, 
                                        StaleElementReferenceException,
                                        NoSuchElementException,
                                        TimeoutException)
import urllib
# from tables_utils import *
from text_processing import *
from table_processing import *
import random
from time import sleep
import csv
import os
import json


def click_with_retry(driver, locator, timeout=60, max_attempts=3, current_url=None):

    attempt = 0
    while attempt < max_attempts:
        try:
            element = WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable(locator)
            )
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center', inline: 'center'});", element)
            sleep(2)  
            element.click()
            WebDriverWait(driver, 60).until(lambda d: d.execute_script('return document.readyState') == 'complete')

            sleep(4)
            return True  # Click successful
        except (ElementClickInterceptedException, StaleElementReferenceException) as e:
            # element = driver.find_element(By.XPATH, locator)
            # element.click()
            attempt += 1
            print(f"Attempt {attempt} failed with {type(e).__name__}. Retrying...")
            if attempt == max_attempts:
                raise
            try:
                element = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(locator)
                )
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
                element.click()
                return True
            except Exception:
                pass 
            sleep(1)  # Brief pause before retry
        except (NoSuchElementException, TimeoutException) as e:
            attempt += 1
            print(f"Attempt {attempt} failed - element not found. Retrying...")
            if attempt == max_attempts:
                raise
            sleep(1)  # Brief pause before retry


def get_element_from_html(html_string, by=By.ID, value='dummy', wait_time=1):
    # driver = get_chrome_driver("dummy_driver")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")

    driver = webdriver.Chrome(executable_path="./chromedriver", options=chrome_options)
 
    html_page = "<html><body><div id='dummy'></div></body></html>"
    encoded_html = urllib.parse.quote(html_page)
    driver.get("data:text/html;charset=utf-8," + encoded_html)
 
    try:
        js_safe_html = json.dumps(html_string)
        driver.execute_script(
            f"document.getElementById('dummy').innerHTML = {js_safe_html};"
        )
        elements = driver.find_elements(by, value)
        if not elements:
            raise ValueError(f"No elements found with {by} = '{value}'")
        return elements[0], driver
    except Exception as e:
        driver.quit()
        raise e
    

def extract_durham_chunks(driver, chunks_title_xpath, chunk_content_xpath):
    chunks_data = []
    
    # all_chunk_titles = driver.find_element(By.XPATH, chunks_title_xpath)
    all_chunk_content = driver.find_elements(By.XPATH, chunk_content_xpath)
    # chunk_content_html = all_chunk_content.get_attribute("outerHTML")
    # all_chunk_title = all_chunk_titles.text
    max_chunk_length = 4000
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_length,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " "]
    )
    
    full_chunk_html = ''
    for chunk_con in all_chunk_content:
        chunk_content_html = chunk_con.get_attribute("outerHTML")
        full_chunk_html = full_chunk_html + "\n" + chunk_content_html

    chunk_element, dummy_driver = get_element_from_html(full_chunk_html)
    final_paragraph_text = table_to_markdown_with_chunks(chunk_element,"nonstandard")
    # final_paragraph_text = final_paragraph_text.replace(all_chunk_title,"")


    if len(final_paragraph_text) > max_chunk_length:
        split_texts = text_splitter.split_text(final_paragraph_text)
        for split_text in split_texts:
            chunks_data.append((chunks_title_xpath, split_text, full_chunk_html))
    else:
        chunks_data.append((chunks_title_xpath, final_paragraph_text, full_chunk_html))
    
    return chunks_data

def save_outputs(data, output_dir, zoneomics_base_url):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV without HTML column
    with open(os.path.join(output_dir, "durham-NC.csv"), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["chapter_no", "heading_1","heading_2","heading_3",
                         "sub_section_no", "source_url", 
                         "zoneomics_url", "text"])
        writer.writerows([row[:-1] for row in data])  # Exclude HTML from CSV
    # Prepare JSON structure
    json_structure = []
    chapter_map = {}
    
    for row in data:
        (chapter_no, heading1, heading2, heading3, sub_sec, url, zoneomics_url, text, html) = row

        file_name = sub_sec.replace("sub-sec-", "")
        file_name = file_name.replace("-", ".")
        
        if chapter_no not in chapter_map:
            chapter_map[chapter_no] = {
                "title": heading1,
                "dir": f"chapter_{chapter_no}",
                "file": file_name if not heading2 else "",
                "is_pdf": "False" if not heading2 else "",
                "sections": []
            }
            json_structure.append(chapter_map[chapter_no])

        chapter = chapter_map[chapter_no]
        
        if heading2:
            sec2 = next((s for s in chapter["sections"] if s["title"] == heading2), None)
            if not sec2:
                sec2 = {
                    "title": heading2,  
                    "file": file_name if not heading3 else "",
                    "is_pdf": "False" if not heading3 else "",
                    "sections": []
                }
                chapter["sections"].append(sec2)
            
            if heading3:
                sec3 = next((s for s in sec2["sections"] if s["title"] == heading3), None)
                if not sec3:
                    sec3 = {
                        "title": heading3, 
                        "file": file_name,
                        "is_pdf": "False",
                        "sections": []
                    }
                    sec2["sections"].append(sec3)
            

        deepest_level = heading3 or heading2 or heading1
        if deepest_level:
            chapter_dir = os.path.join(output_dir, f"chapter_{chapter_no}")
            os.makedirs(chapter_dir, exist_ok=True)
            
            json_data = {
                "source": "durham",
                "title": deepest_level,
                "url": url,
                "html": f"<div id=\"{file_name}\"{html}\n</div>",
            }
            
            with open(os.path.join(chapter_dir, f"{file_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
    

    with open(os.path.join(output_dir, "chapters.json"), 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, indent=4, ensure_ascii=False)
        

# city_name = "spokane-WA"
output_dir = "durham-NC"
zoneomics_base_url = "https://zoneomics.com/code/durham-NC/"
zoning_page_link = "https://udo.durhamnc.gov/udo/Home.htm"

chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")

driver = webdriver.Chrome(executable_path="./chromedriver", options=chrome_options)
actions = ActionChains(driver)

driver.get(zoning_page_link)
driver.maximize_window()
sleep(5)
data = []
first_headings = driver.find_elements(By.XPATH, '//div[@class="sidenav-container"]/ul/li[position()>1]/a')
if first_headings:
    first_heading_data = []
    for heading1 in first_headings:
        # Extract only the direct text of <a> (ignoring nested spans)
        direct_text1 = driver.execute_script(
            "return arguments[0].childNodes[0].textContent.trim();", 
            heading1
        )
        href1 = heading1.get_attribute("href")
        first_heading_data.append((direct_text1, href1))
    # first_headings_href = [(a.text, a.get_attribute("href")) for a in first_headings_1]
    # first_heading_data = [(heading.text, heading.get_attribute("href")) for heading in first_headings]
    for chapter_1, (first_heading_title_text, first_heading) in enumerate(first_heading_data):
        # click_with_retry(driver,first_heading)
        # actions.move_to_element(first_heading).click().perform()
        driver.get(first_heading)

        sleep(5)
        first_heading_text = first_heading_title_text
        chapter_1_idx = chapter_1 + 1
        # if chapter_1_idx > 5:
        #     continue
        subsec_1 = f"sub-sec-{chapter_1_idx}"
        zoneomics_url = f"{zoneomics_base_url}chapter_{chapter_1_idx}"
        
        second_headings = driver.find_elements(By.XPATH, '//div[@class="sidenav-container"]/ul/li/ul/li/a')
        if second_headings:
            second_heading_data = []
            for heading2 in second_headings:
                # Extract only the direct text of <a> (ignoring nested spans)
                direct_text = driver.execute_script(
                    "return arguments[0].childNodes[0].textContent.trim();", 
                    heading2
                )
                href2 = heading2.get_attribute("href")
                second_heading_data.append((direct_text, href2))

            # second_heading_data = [(heading.text, heading.get_attribute("href")) for heading in second_headings]
            for chapter_2, (second_heading_title_text, second_heading) in enumerate(second_heading_data, 1):
                # click_with_retry(driver, second_heading)
                # actions.move_to_element(second_heading).click().perform()
                driver.get(second_heading)

                sleep(5)
                second_heading_text = second_heading_title_text
                subsec_2 = f"{subsec_1}.{chapter_2}"
                chapter_2_idx = f"{chapter_1_idx}.{chapter_2}"
                zoneomics_url_2 = f"{zoneomics_base_url}chapter_{chapter_1_idx}#{chapter_2_idx}"

                third_headings = driver.find_elements(By.XPATH, '//div[@class="sidenav-container"]/ul/li/ul/li//ul/li/a')
                if third_headings:
                    third_heading_data = [(heading.text, heading.get_attribute("href")) for heading in third_headings]
                    for chapter_3, (third_heading_title_text, third_heading) in enumerate(third_heading_data, 1):
                        # click_with_retry(driver,third_heading)
                        # actions.move_to_element(third_heading).click().perform()
                        driver.get(third_heading)


                        sleep(5)
                        third_heading_text = third_heading_title_text
                        
                        subsec_3 = f"{subsec_2}.{chapter_3}"
                        chapter_3_idx = f"{chapter_2_idx}.{chapter_3}"
                        zoneomics_url_3 = f"{zoneomics_base_url}chapter_{chapter_1_idx}#{chapter_3_idx}"

                        chunk_title_text_xpath = third_heading_text
                        whole_chunk_content_xpath = f'//div[@id="mc-main-content"]//h3[@class="Heading3"][1]/following-sibling::*[not(self::h3[@class="Heading3"]) and count(preceding-sibling::h3[@class="Heading3"]) = {chapter_3} ]'
                        chunks = extract_durham_chunks(driver, chunk_title_text_xpath, whole_chunk_content_xpath)
                        for chunk_title, chunk_text, chunk_html in chunks:
                            data.append((chapter_1_idx, first_heading_text, second_heading_text, third_heading_text, subsec_3, driver.current_url, zoneomics_url_3, chunk_text, chunk_html))

                else:
                    chunk_title_text_xpath = second_heading_text
                    whole_chunk_content_xpath = '//div[@id="mc-main-content"]//h2[@class="Heading2"]/following-sibling::*'
                    chunks = extract_durham_chunks(driver, chunk_title_text_xpath, whole_chunk_content_xpath)
                    for chunk_title, chunk_text, chunk_html in chunks:
                        data.append((chapter_1_idx, first_heading_text, second_heading_text, "",subsec_2, driver.current_url, zoneomics_url_2, chunk_text, chunk_html))

                        # chunk_title_full_text = chunk_title_number_text + " " + chunk_title_text
                        # if first_heading_text.lower() == chunk_title_full_text.lower():
                            # whole_chunk_text = whole_chunk_text.replace(chunk_title_full_text, "")
            
                # chunks = extract_durham_chunks(driver, chunk_title_text_xpath, whole_chunk_content_xpath)
                # for chunk_title, chunk_text, chunk_html in chunks:
                #     data.append((chapter_1_idx, first_heading_text, second_heading_text, subsec_2, driver.current_url, zoneomics_url_2, chunk_text, chunk_html))
                # close_dropdown = driver.find_element(By.XPATH, '//div[@class="faq-category"][position() >= 118 and position() <= 166]/div[@class="faq-questions collapse show"]/div[@class="faq-item"][position()>1]//a[@aria-expanded="true"]')
                # actions.move_to_element(close_dropdown).click().perform()
                # # close_dropdown.click()
                # sleep(3)
                
# Save all outputs
save_outputs(data, output_dir, zoneomics_base_url)
print(f"Saved all durham outputs to {output_dir} directory")
driver.quit()