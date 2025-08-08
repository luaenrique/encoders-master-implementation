import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pickle
import os
import time
import sys

def main():
    print("🚀 Iniciando extração de embeddings do Yelp...")
    
    # Configurações
    MODEL_NAMES = [
        'bert-base-uncased',
        'google/electra-base-discriminator', 
        'roberta-base'
    ]
    BATCH_SIZE = 16
    SAVE_DIR = "./embeddings_yelp"  # Diretório local na VM
    
    # Criar diretório se não existir
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 Salvando embeddings em: {SAVE_DIR}")
    
    # Verificar GPU disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU disponível: {gpu_name}")
        print(f"💾 VRAM disponível: {gpu_memory:.1f} GB")
    else:
        print("⚠️  Usando CPU (vai ser mais lento)")
    
    # Carregar dataset Yelp
    print("\n📦 Carregando dataset Yelp...")
    try:
        yelp_dataset = load_dataset("Yelp/yelp_review_full")
        print("✅ Dataset carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        print("💡 Tente: pip install datasets")
        sys.exit(1)
    
    # Dados de treino
    train_texts = yelp_dataset['train']['text']
    train_labels = yelp_dataset['train']['label']
    print(f"📊 Treino - Textos: {len(train_texts):,}")
    
    # Dados de teste
    test_texts = yelp_dataset['test']['text']
    test_labels = yelp_dataset['test']['label']
    print(f"📊 Teste - Textos: {len(test_texts):,}")
    
    # Informações do dataset
    print(f"\n📈 Informações do Yelp:")
    print(f"Classes: {sorted(set(train_labels))} (1-5 estrelas)")
    train_dist = np.bincount(train_labels)
    test_dist = np.bincount(test_labels)
    print(f"Distribuição treino: {train_dist[1:]}")  # Ignorar índice 0
    print(f"Distribuição teste: {test_dist[1:]}")
    
    # Função para extrair embeddings de um batch
    def get_batch_embeddings(batch_texts, model, tokenizer):
        inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden]
            embeddings = last_hidden.mean(dim=1)  # média dos tokens
        
        return embeddings.cpu().numpy()
    
    # Função para processar um split (treino ou teste)
    def process_split(texts, labels, split_name, model, tokenizer, model_name):
        print(f"🔄 Extraindo embeddings {split_name} para {model_name}...")
        all_embeddings = []
        
        progress_bar = tqdm(
            range(0, len(texts), BATCH_SIZE), 
            desc=f"{model_name} [{split_name.upper()}]",
            unit="batch"
        )
        
        for batch_start in progress_bar:
            batch_texts = texts[batch_start:batch_start+BATCH_SIZE]
            batch_embeddings = get_batch_embeddings(batch_texts, model, tokenizer)
            all_embeddings.append(batch_embeddings)
            
            # Mostrar progresso de memória GPU se disponível
            if torch.cuda.is_available() and (batch_start // BATCH_SIZE) % 100 == 0:
                memory_used = torch.cuda.memory_allocated() / 1e9
                progress_bar.set_postfix({'GPU_mem': f'{memory_used:.1f}GB'})
        
        return np.vstack(all_embeddings)
    
    # Extrair embeddings para cada modelo
    all_embeddings = {'train': {}, 'test': {}}
    processing_times = {}
    
    for model_idx, model_name in enumerate(MODEL_NAMES):
        print(f"\n{'='*70}")
        print(f"🤖 Processando modelo {model_idx+1}/{len(MODEL_NAMES)}: {model_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Limpar cache GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Carregar modelo e tokenizer
            print("📥 Carregando modelo e tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            model.to(device)
            
            # Verificar uso de memória
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"💾 Memória GPU após carregar modelo: {memory_used:.2f} GB")
            
            model_key = model_name.replace('/', '_').replace('-', '_')
            
            # Processar treino
            print(f"\n🏋️  Processando dados de TREINO ({len(train_texts):,} exemplos)...")
            X_train = process_split(train_texts, train_labels, 'treino', model, tokenizer, model_name)
            all_embeddings['train'][model_key] = X_train
            
            # Salvar treino
            train_save_path = os.path.join(SAVE_DIR, f"{model_key}_embeddings_train.npy")
            np.save(train_save_path, X_train)
            print(f"💾 Treino salvo: {train_save_path}")
            print(f"📏 Shape treino: {X_train.shape}")
            
            # Processar teste
            print(f"\n🧪 Processando dados de TESTE ({len(test_texts):,} exemplos)...")
            X_test = process_split(test_texts, test_labels, 'teste', model, tokenizer, model_name)
            all_embeddings['test'][model_key] = X_test
            
            # Salvar teste
            test_save_path = os.path.join(SAVE_DIR, f"{model_key}_embeddings_test.npy")
            np.save(test_save_path, X_test)
            print(f"💾 Teste salvo: {test_save_path}")
            print(f"📏 Shape teste: {X_test.shape}")
            
            processing_time = time.time() - start_time
            processing_times[model_name] = processing_time
            
            print(f"\n✅ {model_name} concluído!")
            print(f"   ⏱️  Tempo: {processing_time/60:.1f} minutos")
            print(f"   📊 Treino: {X_train.shape}, Teste: {X_test.shape}")
            
            # Limpar memória
            del model, tokenizer, X_train, X_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Erro ao processar {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Salvar dados completos
    print(f"\n{'='*60}")
    print("💾 Salvando dados completos...")
    
    if all_embeddings['train']:  # Verificar se há dados para salvar
        # Treino
        train_data = {
            'embeddings': all_embeddings['train'],
            'labels': np.array(train_labels),
            'model_names': list(all_embeddings['train'].keys()),
            'dataset_info': {
                'dataset': 'yelp_review_full',
                'split': 'train',
                'num_samples': len(train_labels),
                'embedding_method': 'mean_pooling',
                'max_length': 512
            }
        }
        
        train_pkl_path = os.path.join(SAVE_DIR, 'all_embeddings_yelp_train.pkl')
        with open(train_pkl_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f"✅ Dados de treino salvos: {train_pkl_path}")
        
        # Teste
        test_data = {
            'embeddings': all_embeddings['test'],
            'labels': np.array(test_labels),
            'model_names': list(all_embeddings['test'].keys()),
            'dataset_info': {
                'dataset': 'yelp_review_full',
                'split': 'test',
                'num_samples': len(test_labels),
                'embedding_method': 'mean_pooling',
                'max_length': 512
            }
        }
        
        test_pkl_path = os.path.join(SAVE_DIR, 'all_embeddings_yelp_test.pkl')
        with open(test_pkl_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"✅ Dados de teste salvos: {test_pkl_path}")
        
        # Relatório final
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        print(f"📁 Diretório: {SAVE_DIR}")
        print(f"🤖 Modelos processados: {len(all_embeddings['train'])}")
        print(f"📊 Treino: {len(train_labels):,} amostras")
        print(f"📊 Teste: {len(test_labels):,} amostras")
        
        # Listar arquivos criados
        print(f"\n📄 Arquivos criados:")
        files = sorted(os.listdir(SAVE_DIR))
        for file in files:
            file_path = os.path.join(SAVE_DIR, file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"   {file} ({size_mb:.1f} MB)")
        
        # Tempos de processamento
        if processing_times:
            print(f"\n⏱️  Tempos de processamento:")
            total_time = sum(processing_times.values())
            for model_name, time_sec in processing_times.items():
                print(f"   {model_name}: {time_sec/60:.1f} min")
            print(f"   ⏱️  Tempo total: {total_time/60:.1f} min ({total_time/3600:.1f}h)")
        
        # Instruções de uso
        print(f"\n📖 Para carregar os embeddings:")
        print("```python")
        print("import pickle")
        print("import numpy as np")
        print()
        print("# Carregar dados")
        print(f"with open('{train_pkl_path}', 'rb') as f:")
        print("    train_data = pickle.load(f)")
        print(f"with open('{test_pkl_path}', 'rb') as f:")
        print("    test_data = pickle.load(f)")
        print()
        print("# Extrair embeddings")
        print("X_train_bert = train_data['embeddings']['bert_base_uncased']")
        print("X_test_bert = test_data['embeddings']['bert_base_uncased']")
        print("y_train = train_data['labels']")
        print("y_test = test_data['labels']")
        print("```")
        
    else:
        print("❌ Nenhum modelo foi processado com sucesso!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Processamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)