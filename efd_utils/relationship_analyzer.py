import json
import os
from pathlib import Path
import re
from typing import Dict, List, Set, Tuple

class EFDRelationshipAnalyzer:
    def __init__(self, json_dir: str):
        """
        Inicializa o analisador de relacionamentos.
        
        Args:
            json_dir: Diretório contendo os arquivos JSON dos registros
        """
        self.json_dir = Path(json_dir)
        self.registros: Dict[str, List[Dict]] = {}
        self.relacionamentos: Dict[str, Dict] = {
            'campos': {},  # campo -> registros que o usam
            'registros': {}  # registro -> campos relacionados em outros registros
        }
        self._load_registros()
        
    def _load_registros(self):
        """Carrega todos os registros JSON do diretório."""
        for json_file in self.json_dir.glob('registro_*.json'):
            registro_code = json_file.stem.replace('registro_', '')
            with open(json_file, 'r', encoding='utf-8') as f:
                self.registros[registro_code] = json.load(f)
                
    def find_field_relationships(self) -> Dict[str, Set[str]]:
        """
        Encontra relacionamentos entre campos através de descrições e regras.
        """
        campo_relacionamentos = {}
        
        for reg_code, campos in self.registros.items():
            for campo in campos:
                campo_nome = campo.get('Campo', '')
                descricao = campo.get('Descrição', '').lower()
                
                # Procura referências a outros campos na descrição
                for outro_reg, outros_campos in self.registros.items():
                    if outro_reg != reg_code:
                        for outro_campo in outros_campos:
                            outro_nome = outro_campo.get('Campo', '')
                            if outro_nome and outro_nome.lower() in descricao:
                                # Registra o relacionamento
                                key = f"{reg_code}.{campo_nome}"
                                if key not in campo_relacionamentos:
                                    campo_relacionamentos[key] = set()
                                campo_relacionamentos[key].add(f"{outro_reg}.{outro_nome}")
        
        return campo_relacionamentos
    
    def analyze_field_usage(self) -> Dict[str, Set[str]]:
        """
        Analisa onde cada campo é usado em diferentes registros.
        """
        campo_uso = {}
        
        for reg_code, campos in self.registros.items():
            for campo in campos:
                campo_nome = campo.get('Campo', '')
                if campo_nome:
                    if campo_nome not in campo_uso:
                        campo_uso[campo_nome] = set()
                    campo_uso[campo_nome].add(reg_code)
        
        return campo_uso
    
    def generate_relationship_graph(self, output_file: str):
        """
        Gera um arquivo GraphViz DOT para visualização dos relacionamentos.
        
        Args:
            output_file: Caminho completo para o arquivo .dot de saída
        """
        try:
            relacionamentos = self.find_field_relationships()
            
            dot_content = ['digraph EFD {', 'rankdir=LR;', 'node [shape=box];']
            
            # Adiciona nós para registros
            registros_unicos = set()
            for rel in relacionamentos.items():
                reg_origem = rel[0].split('.')[0]
                registros_unicos.add(reg_origem)
                for destino in rel[1]:
                    reg_destino = destino.split('.')[0]
                    registros_unicos.add(reg_destino)
            
            for reg in registros_unicos:
                dot_content.append(f'  "{reg}" [style=filled,fillcolor=lightblue];')
            
            # Adiciona arestas para relacionamentos
            for origem, destinos in relacionamentos.items():
                for destino in destinos:
                    dot_content.append(f'  "{origem}" -> "{destino}";')
            
            dot_content.append('}')
            
            # Ensure parent directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the DOT file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(dot_content))
                
            print(f"✅ Graph file saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error generating graph: {str(e)}")