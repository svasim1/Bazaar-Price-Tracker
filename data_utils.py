import requests
import json
import time
from datetime import datetime, timedelta


def fetch_all_data(item, start=datetime(2020, 9, 10, 0, 0, 0), end=None, interval_seconds=82800):
    """Fetch bazaar data from API.
    
    Args:
        item: The item ID to fetch
        start: Start datetime for data collection
        end: End datetime for data collection (defaults to now)
        interval_seconds: Interval between API calls
        
    Returns:
        List of data entries from the API
    """
    if end is None:
        end = datetime.now()

    base_url = "https://sky.coflnet.com/api/bazaar"
    interval = timedelta(seconds=interval_seconds)

    current = start
    raw_combined = []

    requests_made = 0
    max_requests = 30
    window_seconds = 10

    while current + interval <= end:
        start_str = current.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
        end_str = (current + interval).strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")

        url = f"{base_url}/{item}/history?start={start_str}&end={end_str}"
        print(f"Fetching: {url}")

        try:
            resp = requests.get(url)
            data = resp.json()

            if isinstance(data, list):
                raw_combined.extend(data)
            elif isinstance(data, dict):
                raw_combined.append(data)

        except Exception as e:
            print(f"Error: {e}")

        requests_made += 1
        if requests_made >= max_requests:
            print(f"Hit {max_requests} requests → waiting {window_seconds} seconds...")
            time.sleep(window_seconds)
            requests_made = 0

        current += interval

    return raw_combined


def load_or_fetch_item_data(item_id, fetch_if_missing=True, update_with_new_data=False):
    """Load item data from file, fetching from API if it doesn't exist.
    
    Args:
        item_id: The item ID to load
        fetch_if_missing: Whether to fetch from API if file doesn't exist
        update_with_new_data: If True and file exists, fetch only new data since last update
        
    Returns:
        List of data entries, or None if file doesn't exist and fetch_if_missing is False
    """
    import os
    
    json_dir = "/Users/samuelbraga/Json Files"
    filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
    
    # Create directory if it doesn't exist
    os.makedirs(json_dir, exist_ok=True)
    
    if not os.path.exists(filename):
        if fetch_if_missing:
            print(f"  → No cache found, fetching full history from API...")
            all_data = fetch_all_data(item_id)
            
            # Save the fetched data
            with open(filename, 'w') as f:
                json.dump(all_data, f, indent=4)
            
            print(f"  ✓ Saved {len(all_data)} entries")
            return all_data
        else:
            print(f"  ✗ File {filename} not found")
            return None
    
    # Load existing data
    print(f"  → Loading from cache...")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Update with new data if requested
    if update_with_new_data and data:
        # Find the most recent timestamp in existing data
        latest_timestamp = None
        for entry in reversed(data):
            if isinstance(entry, dict) and 'timestamp' in entry:
                try:
                    from LGBMfulldata import parse_timestamp
                    latest_timestamp = parse_timestamp(entry['timestamp'])
                    break
                except:
                    continue
        
        if latest_timestamp:
            print(f"  → Fetching new data since {latest_timestamp.strftime('%Y-%m-%d')}...")
            # Fetch data from latest timestamp to now
            new_data = fetch_all_data(item_id, start=latest_timestamp, end=datetime.now())
            
            if new_data:
                # Append new data
                data.extend(new_data)
                
                # Save updated data
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f"  ✓ Added {len(new_data)} new entries (total: {len(data)})")
            else:
                print(f"  ✓ No new data available")
    else:
        print(f"  ✓ Loaded {len(data)} entries")
    
    return data


def delete_item_cache(item_id):
    """Delete cached JSON file for an item to free storage.
    
    Args:
        item_id: The item ID whose cache to delete
    
    Returns:
        bool: True if file was deleted, False if it didn't exist
    """
    import os
    
    json_dir = "/Users/samuelbraga/Json Files"
    filename = os.path.join(json_dir, f"bazaar_history_combined_{item_id}.json")
    
    if os.path.exists(filename):
        try:
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            os.remove(filename)
            print(f"Deleted {filename} ({file_size:.1f} MB)")
            return True
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            return False
    return False


def delete_all_cache():
    """Delete all cached JSON files to free storage.
    
    Returns:
        tuple: (files_deleted, total_mb_freed)
    """
    import os
    
    json_dir = "/Users/samuelbraga/Json Files"
    
    if not os.path.exists(json_dir):
        return 0, 0
    
    files_deleted = 0
    total_bytes = 0
    
    for filename in os.listdir(json_dir):
        if filename.startswith("bazaar_history_combined_") and filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            try:
                total_bytes += os.path.getsize(filepath)
                os.remove(filepath)
                files_deleted += 1
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
    
    total_mb = total_bytes / (1024 * 1024)
    print(f"Deleted {files_deleted} cache files, freed {total_mb:.1f} MB")
    return files_deleted, total_mb


def fetch_recent_data(item_id, hours=24):
    """Fetch only recent data from the API for real-time predictions.
    
    Args:
        item_id: The item ID to fetch
        hours: Number of hours of recent data to fetch (default 24)
        
    Returns:
        List of recent data entries from the API
    """
    end = datetime.now()
    start = end - timedelta(hours=hours)
    
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    
    url = f"https://sky.coflnet.com/api/bazaar/{item_id}/history?start={start_str}&end={end_str}"
    
    try:
        resp = requests.get(url)
        data = resp.json()
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    except Exception as e:
        print(f"Error fetching recent data: {e}")
        return []
