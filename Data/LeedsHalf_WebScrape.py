#!/usr/bin/env python3
"""
Webscraper for chip timing race results from chiptiming.co.uk
Extracts race data and converts to structured CSV format.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChipTimingScraper:
    """Scraper for chip timing race results."""
    
    def __init__(self, base_url: str, delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            base_url: Base URL of the race results page
            delay: Delay between requests to be respectful
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def extract_table_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract data from results tables."""
        results = []
        
        # Look for common table structures
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:  # Skip tables without data
                continue
                
            # Try to identify header row
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Skip if no meaningful headers
            if not any(keyword in ' '.join(headers).lower() 
                      for keyword in ['name', 'time', 'position', 'bib']):
                continue
            
            logger.info(f"Found table with headers: {headers}")
            
            # Extract data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) != len(headers):
                    continue
                    
                row_data = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        row_data[headers[i]] = self._clean_text(cell.get_text(strip=True))
                
                if self._is_valid_result_row(row_data):
                    results.append(row_data)
        
        return results
    
    def extract_list_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract data from list structures (non-table format)."""
        results = []
        
        # Look for common list patterns
        result_containers = soup.find_all(['div', 'li'], class_=re.compile(r'result|runner|participant'))
        
        for container in result_containers:
            result = self._extract_from_container(container)
            if result:
                results.append(result)
        
        return results
    
    def _extract_from_container(self, container) -> Optional[Dict]:
        """Extract result data from a container element."""
        result = {}
        text = container.get_text(strip=True)
        
        # Common patterns for race results
        patterns = {
            'name': re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            'bib': re.compile(r'(?:bib|#)\s*:?\s*(\d+)', re.IGNORECASE),
            'time': re.compile(r'(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})'),
            'position': re.compile(r'(?:pos|position)\s*:?\s*(\d+)', re.IGNORECASE),
            'category': re.compile(r'(?:cat|category)\s*:?\s*([A-Z0-9]+)', re.IGNORECASE)
        }
        
        for field, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                result[field] = match.group(1)
        
        return result if len(result) >= 2 else None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        return re.sub(r'\s+', ' ', text.strip())
    
    def _is_valid_result_row(self, row_data: Dict) -> bool:
        """Check if row contains valid result data."""
        # Must have at least name or bib number and some timing/position info
        has_identifier = any(keyword in ' '.join(row_data.keys()).lower() 
                           for keyword in ['name', 'bib', 'number'])
        has_result = any(keyword in ' '.join(row_data.keys()).lower() 
                        for keyword in ['time', 'position', 'place'])
        
        return has_identifier and has_result
    
    def scrape_results(self) -> List[Dict]:
        """Main scraping method."""
        logger.info(f"Scraping results from {self.base_url}")
        
        soup = self.fetch_page(self.base_url)
        if not soup:
            return []
        
        # Try table extraction first
        results = self.extract_table_data(soup)
        
        # Fallback to list extraction if no table data found
        if not results:
            logger.info("No table data found, trying list extraction")
            results = self.extract_list_data(soup)
        
        # Check for pagination or "Load More" functionality
        results.extend(self._handle_pagination(soup))
        
        logger.info(f"Extracted {len(results)} results")
        return results
    
    def _handle_pagination(self, soup: BeautifulSoup) -> List[Dict]:
        """Handle paginated results."""
        additional_results = []
        
        # Look for pagination links
        pagination_links = soup.find_all('a', href=re.compile(r'page=|p=|\d+'))
        
        for link in pagination_links[:5]:  # Limit to first 5 pages to avoid infinite loops
            href = link.get('href')
            if href:
                time.sleep(self.delay)
                page_url = urljoin(self.base_url, href)
                page_soup = self.fetch_page(page_url)
                
                if page_soup:
                    page_results = self.extract_table_data(page_soup)
                    if not page_results:
                        page_results = self.extract_list_data(page_soup)
                    
                    additional_results.extend(page_results)
        
        return additional_results
    
    def save_to_csv(self, results: List[Dict], output_path: str) -> None:
        """Save results to CSV file."""
        if not results:
            logger.warning("No results to save")
            return
        
        df = pd.DataFrame(results)
        
        # Clean and standardize column names
        df.columns = [self._standardize_column_name(col) for col in df.columns]
        
        # Sort by position if available
        if 'position' in df.columns:
            df['position'] = pd.to_numeric(df['position'], errors='coerce')
            df = df.sort_values('position')
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} results to {output_path}")
    
    def _standardize_column_name(self, name: str) -> str:
        """Standardize column names."""
        name = name.lower().strip()
        
        # Common standardizations
        mappings = {
            'pos': 'position',
            'place': 'position',
            'bib no': 'bib_number',
            'race no': 'bib_number',
            'runner': 'name',
            'participant': 'name',
            'gun time': 'gun_time',
            'chip time': 'chip_time',
            'net time': 'net_time',
            'finish time': 'finish_time',
            'cat': 'category'
        }
        
        return mappings.get(name, name.replace(' ', '_'))
    
    def get_event_info(self) -> Dict:
        """Extract event metadata."""
        soup = self.fetch_page(self.base_url)
        if not soup:
            return {}
        
        event_info = {}
        
        # Extract title
        title = soup.find('title')
        if title:
            event_info['event_name'] = title.get_text(strip=True)
        
        # Extract date and location
        text = soup.get_text()
        date_match = re.search(r'(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})', text)
        if date_match:
            event_info['date'] = date_match.group(1)
        
        location_match = re.search(r'([A-Z][a-z]+,\s*[A-Z][a-z]+)', text)
        if location_match:
            event_info['location'] = location_match.group(1)
        
        return event_info


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Scrape chip timing race results')
    parser.add_argument('url', help='URL of the race results page')
    parser.add_argument('-o', '--output', default='race_results.csv', help='Output CSV filename')
    parser.add_argument('-d', '--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    scraper = ChipTimingScraper(args.url, args.delay)
    
    # Get event information
    event_info = scraper.get_event_info()
    if event_info:
        logger.info(f"Event: {event_info}")
    
    # Scrape results
    results = scraper.scrape_results()
    
    if results:
        # Save to CSV
        output_path = Path(args.output)
        scraper.save_to_csv(results, str(output_path))
        
        # Display summary
        df = pd.DataFrame(results)
        print(f"\nScraping Summary:")
        print(f"Total results: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Output saved to: {output_path}")
        
        # Show first few results
        print(f"\nFirst 5 results:")
        print(df.head().to_string(index=False))
    else:
        print("No results found. The event may not have occurred yet or the page structure may have changed.")


if __name__ == "__main__":
    main()