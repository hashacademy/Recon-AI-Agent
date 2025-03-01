import os
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from IPython.display import Image, display
import xml.etree.ElementTree as ET
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Annotated
from operator import add
from langgraph.graph.message import add_messages

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI( 
    model="gpt-3.5-turbo",
    temperature=0.7,
    timeout=60
)


@dataclass
class ReconState:
    target: str
    subdomains: Annotated[list[str], add]
    messages: Annotated[list, add_messages]
    nmap_result: Annotated[dict, add]
    # report: str


# tool to run any command
def run_command(command, arguments, capture_output=False, check=False, shell=False):
    """
    Execute a command with arguments.

    Parameters:
        command (str): The command to execute.
        arguments (list): List of arguments for the command.
        capture_output (bool): If True, captures stdout and stderr.
        check (bool): If True, raises CalledProcessError on non-zero exit code.
        shell (bool): If True, executes the command through the shell.

    Returns:
        subprocess.CompletedProcess: Result of the command execution.
    """
    # Combine command and arguments into a single list
    cmd = [command] + arguments
    
    # Determine stdout and stderr handling
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    
    # Execute the command
    result = subprocess.run(
        cmd,
        stdout=stdout,
        stderr=stderr,
        text=True,
        check=check,
        shell=shell
    )
    
    return result

def parse_nmap_xml(file_path: str):
    """
    Parse Nmap XML output and return structured data as a dictionary.
    
    Args:
        file_path (str): Path to Nmap XML output file
    
    Returns:
        dict: Structured Nmap scan results
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    results = {
        'hosts': [],
        'scan_info': {
            'scanner': root.attrib.get('scanner'),
            'args': root.attrib.get('args'),
            'start': root.attrib.get('start'),
            'version': root.attrib.get('version')
        }
    }

    for host in root.findall('host'):
        host_info = {
            'status': {},
            'addresses': [],
            'hostnames': [],
            'ports': [],
            'os': {},
            'scripts': []
        }

        # Host status
        status = host.find('status')
        if status is not None:
            host_info['status'] = {
                'state': status.attrib.get('state'),
                'reason': status.attrib.get('reason')
            }

        # IP addresses and MAC
        for address in host.findall('address'):
            addr_type = address.attrib.get('addrtype')
            host_info['addresses'].append({
                'type': addr_type,
                'addr': address.attrib.get('addr'),
                'vendor': address.attrib.get('vendor', '')
            })

        # Hostnames
        for hostname in host.findall('hostnames/hostname'):
            host_info['hostnames'].append({
                'name': hostname.attrib.get('name'),
                'type': hostname.attrib.get('type')
            })

        # Ports and services
        ports = host.find('ports')
        if ports is not None:
            for port in ports.findall('port'):
                port_info = {
                    'port': port.attrib.get('portid'),
                    'protocol': port.attrib.get('protocol'),
                    'state': {},
                    'service': {},
                    'scripts': []
                }

                # Port state
                state = port.find('state')
                if state is not None:
                    port_info['state'] = {
                        'state': state.attrib.get('state'),
                        'reason': state.attrib.get('reason')
                    }

                # Service information
                service = port.find('service')
                if service is not None:
                    port_info['service'] = {
                        'name': service.attrib.get('name'),
                        'product': service.attrib.get('product', ''),
                        'version': service.attrib.get('version', ''),
                        'extrainfo': service.attrib.get('extrainfo', ''),
                        'ostype': service.attrib.get('ostype', ''),
                        'method': service.attrib.get('method'),
                        'conf': service.attrib.get('conf')
                    }

                # Script output
                for script in port.findall('script'):
                    port_info['scripts'].append({
                        'id': script.attrib.get('id'),
                        'output': script.attrib.get('output'),
                        'tables': [tab.attrib for tab in script.findall('table')]
                    })

                host_info['ports'].append(port_info)

        # OS detection
        os = host.find('os')
        if os is not None:
            os_matches = []
            for match in os.findall('osmatch'):
                os_matches.append({
                    'name': match.attrib.get('name'),
                    'accuracy': match.attrib.get('accuracy'),
                    'line': match.attrib.get('line')
                })
            
            os_classes = []
            for cls in os.findall('osclass'):
                os_classes.append({
                    'type': cls.attrib.get('type'),
                    'vendor': cls.attrib.get('vendor'),
                    'osfamily': cls.attrib.get('osfamily'),
                    'osgen': cls.attrib.get('osgen'),
                    'accuracy': cls.attrib.get('accuracy')
                })

            host_info['os'] = {
                'matches': os_matches,
                'classes': os_classes
            }

        # Host scripts
        for script in host.findall('hostscript/script'):
            host_info['scripts'].append({
                'id': script.attrib.get('id'),
                'output': script.attrib.get('output'),
                'tables': [tab.attrib for tab in script.findall('table')]
            })

        results['hosts'].append(host_info)
    
    return results

def is_tool_installed(tool_name: str):
    """Check if tool installed"""
    try:
        subprocess.run(
            ["/usr/bin/which", tool_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

agent_tools = [run_command, parse_nmap_xml, is_tool_installed]
llm_with_tools = llm.bind_tools(tools=agent_tools)

def chatbot(state: ReconState):
    return {"messages": [llm_with_tools.invoke(state.messages)]}


graph_builder = StateGraph(ReconState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(agent_tools))
graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

display(graph.get_graph().draw_ascii())

result = graph.invoke({"messages": "You are a professional cybersecurity engineer. Your manager demanded from you to perfrom recon on your company. You have a list of handy cybersecurity recon tools on your toolbox. Here is the domain of the company google.com. Your output should be a markdown report contains all the identified assets.", "target":"google.com"})

# print(result)
for m in result['messages']:
	m.pretty_print()
# print(result)
# Execute recon
# result = graph.invoke(ReconState(target="facebook.com"))