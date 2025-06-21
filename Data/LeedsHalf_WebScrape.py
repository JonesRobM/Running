#!/usr/bin/env python3
"""
Enhanced webscraper for chip timing race results from chiptiming.co.uk
Handles JavaScript-loaded content and multiple data formats.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
import logging
from urllib.parse import urljoin, urlparse, parse_qs
from typing import List, Dict, Optional, Union
import argparse
from pathlib import Path

# Try to import selenium for JavaScript rendering
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChipTimingScraper:
    """Enhanced scraper for chip timing race results."""
    
    def __init__(self, base_url: str, delay: float = 1.0, use_selenium: bool = True):
        """
        Initialize scraper.
        
        Args:
            base_url: Base URL of the race results page
            delay: Delay between requests
            use_selenium: Whether to use Selenium for JavaScript content
        """
        self.base_url = base_url
        self.delay = delay
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-GB,en;q=0.9',
            'Referer': base_url
        })
        
        if not SELENIUM_AVAILABLE and use_selenium:
            logger.warning("Selenium not available. Install with: pip install selenium")
    
    def setup_driver(self) -> Optional[webdriver.Chrome]:
        """Setup Chrome driver for Selenium."""
        if not self.use_selenium:
            return None
            
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument(f'--user-agent={self.session.headers["User-Agent"]}')
            
            driver = webdriver.Chrome(options=options)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return None
    
    def fetch_page_selenium(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch page content using Selenium."""
        driver = self.setup_driver()
        if not driver:
            return None
        
        try:
            driver.get(url)
            
            # Wait for results to load (look for common indicators)
            wait = WebDriverWait(driver, 15)
            
            # Try multiple selectors that might indicate loaded content
            selectors_to_try = [
                "table",
                ".result",
                ".runner",
                ".participant",
                "[class*='result']",
                "[id*='result']",
                "tbody tr",
                ".data-table"
            ]
            
            content_loaded = False
            for selector in selectors_to_try:
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    content_loaded = True
                    logger.info(f"Content loaded using selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not content_loaded:
                logger.info("No specific content selectors found, waiting for general page load")
                time.sleep(5)  # Give extra time for any delayed loading
            
            # Get page source after JavaScript execution
            html = driver.page_source
            return BeautifulSoup(html, 'html.parser')
            
        except Exception as e:
            logger.error(f"Selenium error: {e}")
            return None
        finally:
            driver.quit()
    
    def fetch_page_requests(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch page using requests (fallback)."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def discover_api_endpoints(self, soup: BeautifulSoup) -> List[str]:
        """Try to discover AJAX/API endpoints for data."""
        endpoints = []
        
        # Look for JavaScript files that might contain API calls
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                # Look for common API patterns
                api_patterns = [
                    r'fetch\([\'"]([^\'"]*/api/[^\'"]*)[\'"]',
                    r'ajax\s*:\s*[\'"]([^\'"]*/[^\'"]*)[\'"]',
                    r'url\s*:\s*[\'"]([^\'"]*/[^\'"]*(?:json|data)[^\'"]*)[\'"]',
                    r'[\'"]([^\'"]*/results[^\'"]*)[\'"]'
                ]
                
                for pattern in api_patterns:
                    matches = re.findall(pattern, script.string)
                    for match in matches:
                        if not match.startswith('http'):
                            match = urljoin(self.base_url, match)
                        endpoints.append(match)
        
        return list(set(endpoints))  # Remove duplicates
    
    def try_api_endpoints(self, endpoints: List[str]) -> List[Dict]:
        """Try discovered API endpoints."""
        results = []
        
        for endpoint in endpoints:
            try:
                logger.info(f"Trying API endpoint: {endpoint}")
                response = self.session.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if isinstance(data, list):
                            results.extend(data)
                        elif isinstance(data, dict) and 'results' in data:
                            results.extend(data['results'])
                        elif isinstance(data, dict) and 'data' in data:
                            results.extend(data['data'])
                    except json.JSONDecodeError:
                        # Try to parse as CSV or other format
                        content = response.text
                        if ',' in content and '\n' in content:
                            lines = content.strip().split('\n')
                            if len(lines) > 1:
                                headers = lines[0].split(',')
                                for line in lines[1:]:
                                    values = line.split(',')
                                    if len(values) == len(headers):
                                        results.append(dict(zip(headers, values)))
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.debug(f"API endpoint {endpoint} failed: {e}")
                continue
        
        return results
    
    def extract_table_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Enhanced table data extraction."""
        results = []
        
        # More comprehensive table selectors
        table_selectors = [
            'table',
            '.results-table',
            '.data-table',
            '[class*="table"]',
            '[id*="table"]',
            '[class*="result"]',
            'tbody'
        ]
        
        for selector in table_selectors:
            tables = soup.select(selector)
            
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) < 2:
                    continue
                
                # More flexible header detection
                header_candidates = []
                for i, row in enumerate(rows[:3]):  # Check first 3 rows
                    cells = row.find_all(['th', 'td'])
                    cell_texts = [cell.get_text(strip=True).lower() for cell in cells]
                    
                    # Score potential headers
                    header_score = sum(1 for text in cell_texts 
                                     if any(keyword in text for keyword in 
                                           ['name', 'time', 'position', 'place', 'bib', 'number', 'category', 'club']))
                    
                    if header_score >= 2:  # Need at least 2 relevant columns
                        header_candidates.append((i, row, header_score))
                
                if not header_candidates:
                    continue
                
                # Use best header row
                header_idx, header_row, _ = max(header_candidates, key=lambda x: x[2])
                headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
                
                logger.info(f"Found table with headers: {headers}")
                
                # Extract data rows (skip header and any rows before it)
                for row in rows[header_idx + 1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < len(headers):
                        continue
                    
                    row_data = {}
                    for i, cell in enumerate(cells[:len(headers)]):
                        text = self._clean_text(cell.get_text(strip=True))
                        if text and i < len(headers):
                            row_data[headers[i]] = text
                    
                    if self._is_valid_result_row(row_data):
                        results.append(row_data)
        
        return results
    
    def extract_structured_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract structured data from JSON-LD or other formats."""
        results = []
        
        # Look for JSON-LD structured data
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and 'sportingEvent' in str(data):
                    # Process sporting event data
                    pass
            except json.JSONDecodeError:
                continue
        
        return results
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning."""
        if not text:
            return ""
        
        # Remove common artifacts
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s:.-]', '', text)  # Keep basic punctuation
        return text
    
    def _get_total_pages(self, soup: BeautifulSoup) -> int:
        """Extract total number of pages from pagination info."""
        # Look for "Page X of Y" pattern
        page_text = soup.get_text()
        page_match = re.search(r'Page\s+\d+\s+of\s+(\d+)', page_text, re.IGNORECASE)
        if page_match:
            return int(page_match.group(1))
        
        # Look for pagination links
        pagination_links = soup.find_all('a', href=True)
        page_numbers = []
        
        for link in pagination_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Extract page numbers from href or text
            page_match = re.search(r'page[=:](\d+)', href)
            if page_match:
                page_numbers.append(int(page_match.group(1)))
            elif text.isdigit():
                page_numbers.append(int(text))
        
        return max(page_numbers) if page_numbers else 1
    
    def _is_valid_result_row(self, row_data: Dict) -> bool:
        """Enhanced validation for result rows."""
        if not row_data:
            return False
        
        # Convert keys to lowercase for checking
        keys_lower = [k.lower() for k in row_data.keys()]
        values = list(row_data.values())
        
        # Must have some identifying information
        has_identifier = any(keyword in ' '.join(keys_lower) 
                           for keyword in ['name', 'bib', 'number', 'runner', 'participant'])
        
        # Must have some result information
        has_result = any(keyword in ' '.join(keys_lower) 
                        for keyword in ['time', 'position', 'place', 'rank'])
        
        # Check for time format in values
        has_time_format = any(re.match(r'\d{1,2}:\d{2}(:\d{2})?', str(v)) for v in values)
        
        # Must have at least 2 non-empty values
        has_content = len([v for v in values if v and str(v).strip()]) >= 2
        
        return has_content and (has_identifier or has_result or has_time_format)
    
    def scrape_results(self, max_pages: Optional[int] = None) -> List[Dict]:
        """Enhanced main scraping method with comprehensive pagination."""
        logger.info(f"Scraping results from {self.base_url}")
        
        # Try Selenium first if available
        if self.use_selenium:
            logger.info("Using Selenium for JavaScript content")
            soup = self.fetch_page_selenium(self.base_url)
        else:
            logger.info("Using requests (Selenium not available)")
            soup = self.fetch_page_requests(self.base_url)
        
        if not soup:
            return []
        
        results = []
        
        # First, get results from the initial page
        initial_results = self.extract_table_data(soup)
        if initial_results:
            logger.info(f"Found {len(initial_results)} results on initial page")
            results.extend(initial_results)
        
        # Check if there are multiple pages
        total_pages = self._get_total_pages(soup)
        
        if total_pages > 1:
            if max_pages:
                total_pages = min(total_pages, max_pages)
                logger.info(f"Limiting to {max_pages} pages (out of {self._get_total_pages(soup)} total)")
            
            logger.info(f"Detected {total_pages} pages. Starting comprehensive pagination...")
            
            # Handle pagination starting from page 2 (since we already got page 1)
            if total_pages > 1:
                paginated_results = self._handle_pagination_from_page_2(soup, total_pages)
                results.extend(paginated_results)
        
        # Try other extraction methods if still no results
        if not results:
            extraction_methods = [
                ("structured data", self.extract_structured_data),
            ]
            
            for method_name, method in extraction_methods:
                logger.info(f"Trying {method_name}")
                method_results = method(soup)
                if method_results:
                    logger.info(f"Found {len(method_results)} results using {method_name}")
                    results.extend(method_results)
        
        # Try API endpoints as last resort
        if not results:
            logger.info("Trying to discover API endpoints")
            endpoints = self.discover_api_endpoints(soup)
            if endpoints:
                api_results = self.try_api_endpoints(endpoints)
                results.extend(api_results)
        
        # Remove duplicates
        if results:
            seen = set()
            unique_results = []
            for result in results:
                # Create a key from important fields to detect duplicates
                key_fields = ['name', 'bib_number', 'finish_time', 'position']
                key_values = []
                
                for field in key_fields:
                    value = None
                    # Try different possible column names
                    for col in result.keys():
                        if field in col.lower() or col.lower() in field:
                            value = result[col]
                            break
                    key_values.append(str(value) if value else "")
                
                result_key = "|".join(key_values)
                
                if result_key not in seen:
                    seen.add(result_key)
                    unique_results.append(result)
            
            results = unique_results
            logger.info(f"After deduplication: {len(results)} unique results")
        
        logger.info(f"Total extracted results: {len(results)}")
        return results
    
    def save_to_csv(self, results: List[Dict], output_path: str) -> None:
        """Enhanced CSV saving with better column handling."""
        if not results:
            logger.warning("No results to save")
            return
        
        df = pd.DataFrame(results)
        
        # Clean and standardize column names
        df.columns = [self._standardize_column_name(col) for col in df.columns]
        
        # Clean data
        for col in df.columns:
            if col in ['position', 'place', 'rank', 'bib_number']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by position if available
        sort_columns = ['position', 'place', 'rank']
        for sort_col in sort_columns:
            if sort_col in df.columns and not df[sort_col].isna().all():
                df = df.sort_values(sort_col)
                break
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} results to {output_path}")
        
        # Show summary
        print(f"\nData Summary:")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        if not df.empty:
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string(index=False))
    
    def _standardize_column_name(self, name: str) -> str:
        """Enhanced column name standardization."""
        name = str(name).lower().strip()
        
        # Common standardizations
        mappings = {
            'pos': 'position',
            'place': 'position', 
            'rank': 'position',
            'bib no': 'bib_number',
            'bib': 'bib_number',
            'race no': 'bib_number',
            'number': 'bib_number',
            'runner': 'name',
            'participant': 'name',
            'athlete': 'name',
            'gun time': 'gun_time',
            'chip time': 'chip_time',
            'net time': 'net_time',
            'finish time': 'finish_time',
            'time': 'finish_time',
            'cat': 'category',
            'age cat': 'age_category',
            'gender': 'sex',
            'team': 'club'
        }
        
        # Apply mappings
        standardized = mappings.get(name, name)
        
        # Clean up the name
        standardized = re.sub(r'[^\w\s]', '', standardized)
        standardized = re.sub(r'\s+', '_', standardized)
        
        return standardized
    
    def get_event_info(self) -> Dict:
        """Enhanced event info extraction."""
        if self.use_selenium:
            soup = self.fetch_page_selenium(self.base_url)
        else:
            soup = self.fetch_page_requests(self.base_url)
            
        if not soup:
            return {}
        
        event_info = {}
        
        # Extract title
        title = soup.find('title')
        if title:
            event_info['event_name'] = title.get_text(strip=True)
        
        # Extract from page text
        text = soup.get_text()
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                event_info['date'] = match.group(1)
                break
        
        # Location patterns
        location_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',
            r'Location[:\s]+([^,\n]+)',
            r'Venue[:\s]+([^,\n]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                event_info['location'] = match.group(1).strip()
                break
        
        return event_info


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Enhanced chip timing race results scraper')
    parser.add_argument('url', help='URL of the race results page')
    parser.add_argument('-o', '--output', default='race_results.csv', help='Output CSV filename')
    parser.add_argument('-d', '--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--no-selenium', action='store_true', help='Disable Selenium (use requests only)')
    parser.add_argument('--max-pages', type=int, help='Maximum number of pages to scrape (default: all pages)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    use_selenium = not args.no_selenium
    scraper = ChipTimingScraper(args.url, args.delay, use_selenium)
    
    # Get event information
    event_info = scraper.get_event_info()
    if event_info:
        logger.info(f"Event: {event_info}")
    
    # Add time tracking
    start_time = time.time()
    
    # Scrape results
    results = scraper.scrape_results(max_pages=args.max_pages)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if results:
        # Save to CSV
        output_path = Path(args.output)
        scraper.save_to_csv(results, str(output_path))
        
        print(f"\n{'='*50}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*50}")
        print(f"Total results: {len(results)}")
        print(f"Time taken: {duration:.1f} seconds")
        print(f"Average time per result: {duration/len(results):.3f} seconds")
        print(f"Output saved to: {output_path}")
        
        # Show data quality summary
        df = pd.DataFrame(results)
        print(f"\nData Quality Summary:")
        print(f"Columns: {list(df.columns)}")
        print(f"Completeness:")
        for col in df.columns:
            non_empty = df[col].notna().sum()
            percentage = (non_empty / len(df)) * 100
            print(f"  {col}: {non_empty}/{len(df)} ({percentage:.1f}%)")
        
    else:
        print("No results found.")
        print("\nTroubleshooting tips:")
        print("1. Check if the event has occurred and results are published")
        print("2. Install Selenium for JavaScript support: pip install selenium")
        print("3. Ensure Chrome browser is installed for Selenium")
        print("4. Try the --verbose flag for more detailed logging")
        print("5. Test with a working URL from a past event first")


if __name__ == "__main__":
    main()