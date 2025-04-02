import json
import re


def parse_log_data(file_path, save_json=False):
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    result = {
        'configs': [],
        'dhcp': {},
        'measurements': [],
        'total_utilisation': [],
        'pd_utilisation': {}
    }
    
    current_dict = None
    in_block = False
    
    for line in (line_data := [line.strip() for line in lines]):
        if line.endswith('defined'):
            result['configs'].append(line.split()[0])
            continue
        
        if dhcp_match := re.match(r'LWIP\|NOTICE: DHCP request for (client\d+) returned IP address: (\d+\.\d+\.\d+\.\d+)', line):
            client, ip = dhcp_match.groups()
            result['dhcp'][client] = ip
            continue
        
        if line == 'client0 measurement finished':
            current_dict = {}
            result['measurements'].append(current_dict)
            in_block = True
            continue
        
        if line == 'Total utilisation details:':
            current_dict = {}
            result['total_utilisation'].append(current_dict)
            in_block = True
            continue
        
        if pd_match := re.match(r'Utilisation details for PD: (\w+) \((\d+)\)', line):
            pd_name, pd_id = pd_match.groups()
            current_pd = f"{pd_name} ({pd_id})"
            result['pd_utilisation'].setdefault(current_pd, []).append(current_dict := {})
            in_block = True
            continue
        
        if in_block and (kv_match := re.match(r'(\w+[-\s\w]*):\s*([0-9A-Fa-f]+)', line)):
            key, value = kv_match.groups()
            current_dict[key.replace(' ', '_')] = value
            continue
        
        if line == '}':
            in_block = False
            continue
    
    if save_json:
        with open(f"{file_path}.json", 'w') as f:
            json.dump(result, f, indent=4)
    
    return result
