# dimensionality_reduction_topic_modeling.py
import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, PCA, NMF
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import json

try:
    from umap import UMAP
except ImportError:
    try:
        from umap.umap_ import UMAP
    except ImportError:
        UMAP = None

class DimensionalityReduction:
    def __init__(self):
        self.models = {}
        self.explained_variances = {}
        self.vector_matrix = None
        self.feature_names = None
        self.dense_matrix = None
        
    def load_vectors(self, vector_matrix, feature_names=None):
        """Загрузка векторного представления текстов"""
        self.vector_matrix = vector_matrix
        self.feature_names = feature_names
        
        if issparse(vector_matrix):
            self.dense_matrix = vector_matrix.toarray()
        else:
            self.dense_matrix = vector_matrix
            
        return self.dense_matrix.shape

    def apply_pca(self, n_components=100, random_state=42):
        """Применение PCA для снижения размерности"""
        max_possible_components = min(self.dense_matrix.shape) - 1
        n_components = min(n_components, max_possible_components)
        
        pca = PCA(
            n_components=n_components,
            random_state=random_state
        )
        
        pca_transformed = pca.fit_transform(self.dense_matrix)
        self.models['pca'] = pca
        self.explained_variances['pca'] = pca.explained_variance_ratio_
        
        return pca_transformed
    
    def apply_svd(self, n_components=100, random_state=42):
        """Применение SVD (LSA) для снижения размерности"""
        max_possible_components = min(self.dense_matrix.shape) - 1
        n_components = min(n_components, max_possible_components)
        
        svd = TruncatedSVD(
            n_components=n_components,
            random_state=random_state
        )
        
        if issparse(self.vector_matrix):
            svd_transformed = svd.fit_transform(self.vector_matrix)
        else:
            svd_transformed = svd.fit_transform(self.dense_matrix)
            
        self.models['svd'] = svd
        self.explained_variances['svd'] = svd.explained_variance_ratio_
        
        return svd_transformed
    
    def apply_tsne(self, n_components=2, random_state=42, **kwargs):
        """Применение t-SNE для визуализации"""
        perplexity = kwargs.get('perplexity', min(30, self.dense_matrix.shape[0] - 1))
        
        tsne = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            learning_rate=kwargs.get('learning_rate', 200)
        )
        
        tsne_transformed = tsne.fit_transform(self.dense_matrix)
        self.models['tsne'] = tsne
        
        return tsne_transformed
    
    def apply_lda(self, n_components=10, random_state=42, **kwargs):
        """Тематическое моделирование с помощью LDA"""
        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        
        lda_transformed = lda.fit_transform(self.vector_matrix)
        self.models['lda'] = lda
        
        # Получаем ключевые слова для тем
        if self.feature_names is not None:
            topics = self.interpret_lda_components(n_top_words=10)
        else:
            topics = [f"Тема {i+1}" for i in range(n_components)]
            
        return lda_transformed, topics
    
    def apply_nmf(self, n_components=10, random_state=42, **kwargs):
        """Применение NMF для тематического моделирования"""
        nmf = NMF(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        
        nmf_transformed = nmf.fit_transform(self.dense_matrix)
        self.models['nmf'] = nmf
        
        # Получаем ключевые слова для тем
        if self.feature_names is not None:
            topics = self.interpret_nmf_components(n_top_words=10)
        else:
            topics = [f"Тема {i+1}" for i in range(n_components)]
            
        return nmf_transformed, topics
    
    def interpret_nmf_components(self, n_top_words=10):
        """Интерпретация NMF тем через ключевые слова"""
        if 'nmf' not in self.models or self.feature_names is None:
            return None
            
        nmf = self.models['nmf']
        components = nmf.components_
        
        topic_keywords = []
        for i, component in enumerate(components):
            top_indices = component.argsort()[-n_top_words:][::-1]
            top_words = [self.feature_names[idx] for idx in top_indices]
            top_weights = [component[idx] for idx in top_indices]
            
            topic_keywords.append({
                'topic': i,
                'keywords': list(zip(top_words, top_weights))
            })
            
        return topic_keywords
    
    def find_optimal_components(self, max_components=200, variance_threshold=0.95):
        """Поиск оптимального числа компонент для заданного порога дисперсии"""
        max_components = min(max_components, min(self.dense_matrix.shape) - 1)
        
        svd = TruncatedSVD(n_components=max_components, random_state=42)
        svd.fit(self.vector_matrix)
        
        cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
        optimal_idx = np.where(cumulative_variance >= variance_threshold)[0]
        
        if len(optimal_idx) > 0:
            optimal_components = optimal_idx[0] + 1
            achieved_variance = cumulative_variance[optimal_components - 1]
        else:
            optimal_components = max_components
            achieved_variance = cumulative_variance[-1]
        
        return {
            'optimal_components': optimal_components,
            'achieved_variance': achieved_variance,
            'cumulative_variance': cumulative_variance,
            'components_range': range(1, max_components + 1),
            'variance_threshold': variance_threshold
        }
    
    def visualize_components(self, reduced_matrix, labels=None, method='tsne', 
                           n_components=2, random_state=42, **kwargs):
        """Визуализация компонент с помощью t-SNE или UMAP"""
        if reduced_matrix.shape[1] > n_components:
            if method == 'tsne':
                perplexity = kwargs.get('perplexity', min(30, reduced_matrix.shape[0] - 1))
                viz_model = TSNE(
                    n_components=n_components,
                    random_state=random_state,
                    perplexity=perplexity,
                    learning_rate=kwargs.get('learning_rate', 200)
                )
            elif method == 'umap':
                if UMAP is None:
                    raise ImportError("UMAP не установлен")
                viz_model = UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=kwargs.get('n_neighbors', 15),
                    min_dist=kwargs.get('min_dist', 0.1)
                )
            else:
                raise ValueError("Method must be 'tsne' or 'umap'")
                
            visualization = viz_model.fit_transform(reduced_matrix)
        else:
            visualization = reduced_matrix
            
        return visualization
    
    def create_variance_plot(self, optimal_result, save_path=None):
        """Создание графика кумулятивной дисперсии"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(optimal_result['components_range'], 
                optimal_result['cumulative_variance'], 
                'b-', linewidth=2)
        ax1.axhline(y=optimal_result['variance_threshold'], 
                   color='r', linestyle='--', 
                   label=f'Порог ({optimal_result["variance_threshold"]})')
        ax1.axvline(x=optimal_result['optimal_components'], 
                   color='g', linestyle='--',
                   label=f'Оптимум ({optimal_result["optimal_components"]})')
        ax1.set_xlabel('Число компонент')
        ax1.set_ylabel('Кумулятивная дисперсия')
        ax1.set_title('Кумулятивная дисперсия vs Число компонент')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        individual_variance = np.diff(optimal_result['cumulative_variance'], prepend=0)
        ax2.plot(optimal_result['components_range'][:50], 
                individual_variance[:50], 'g-', linewidth=2)
        ax2.set_xlabel('Число компонент')
        ax2.set_ylabel('Индивидуальная дисперсия')
        ax2.set_title('Индивидуальная дисперсия (первые 50 компонент)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_visualization_plot(self, visualization, labels=None, save_path=None):
        """Создание графика визуализации"""
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                plt.scatter(visualization[mask, 0], visualization[mask, 1], 
                           label=label, alpha=0.7, s=30)
            plt.legend()
        else:
            plt.scatter(visualization[:, 0], visualization[:, 1], alpha=0.7, s=30)
        
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        plt.title('Визуализация сниженной размерности')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def interpret_svd_components(self, n_top_words=10, n_components=None):
        """Интерпретация SVD компонент через ключевые слова"""
        if 'svd' not in self.models or self.feature_names is None:
            return None
        
        svd = self.models['svd']
        components = svd.components_
        
        if n_components is not None:
            components = components[:n_components]
        
        topic_keywords = []
        for i, component in enumerate(components):
            top_indices = component.argsort()[-n_top_words:][::-1]
            top_words = [self.feature_names[idx] for idx in top_indices]
            top_weights = [component[idx] for idx in top_indices]
            
            topic_keywords.append({
                'component': i,
                'explained_variance': svd.explained_variance_ratio_[i],
                'keywords': list(zip(top_words, top_weights))
            })
            
        return topic_keywords
    
    def interpret_lda_components(self, n_top_words=10):
        """Интерпретация LDA тем через ключевые слова"""
        if 'lda' not in self.models or self.feature_names is None:
            return None
            
        lda = self.models['lda']
        components = lda.components_
        
        topic_keywords = []
        for i, component in enumerate(components):
            top_indices = component.argsort()[-n_top_words:][::-1]
            top_words = [self.feature_names[idx] for idx in top_indices]
            top_weights = [component[idx] for idx in top_indices]
            
            topic_keywords.append({
                'topic': i,
                'keywords': list(zip(top_words, top_weights))
            })
            
        return topic_keywords
    
    def analyze_component_quality(self, n_components_range, ground_truth=None):
        """Анализ зависимости качества от числа компонент"""
        results = []
        
        for n_comp in n_components_range:
            if n_comp > min(self.dense_matrix.shape):
                continue
                
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            transformed = svd.fit_transform(self.vector_matrix)
            
            explained_variance = svd.explained_variance_ratio_.sum()
            
            reconstructed = svd.inverse_transform(transformed)
            if issparse(self.vector_matrix):
                original = self.vector_matrix.toarray()
            else:
                original = self.dense_matrix
            reconstruction_error = np.mean((original - reconstructed) ** 2)
            
            silhouette = None
            if ground_truth is not None and len(np.unique(ground_truth)) > 1:
                try:
                    silhouette = silhouette_score(transformed, ground_truth)
                except:
                    silhouette = None
            
            results.append({
                'n_components': n_comp,
                'explained_variance': explained_variance,
                'reconstruction_error': reconstruction_error,
                'silhouette_score': silhouette,
                'matrix_density': np.count_nonzero(transformed) / transformed.size
            })
            
        return results
    
    def get_component_statistics(self, reduced_matrix):
        """Получение статистики по компонентам"""
        if reduced_matrix is None:
            return None
            
        return {
            'original_dimensions': self.dense_matrix.shape,
            'reduced_dimensions': reduced_matrix.shape,
            'compression_ratio': self.dense_matrix.shape[1] / reduced_matrix.shape[1],
            'component_means': np.mean(reduced_matrix, axis=0).tolist(),
            'component_stds': np.std(reduced_matrix, axis=0).tolist(),
            'sparsity_ratio': 1.0 - (np.count_nonzero(reduced_matrix) / reduced_matrix.size)
        }
    
    def save_results(self, filepath, results):
        """Сохранение результатов в JSON"""
        try:
            def convert_numpy_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            results_serializable = convert_numpy_types(results)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception:
            return False