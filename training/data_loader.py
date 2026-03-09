from whistress.training.processor import DSProcessor
from datasets import load_from_disk
import os
from datasets import Dataset, Audio
import scipy.io as sio
from pathlib import Path

class PreprocessedDataLoader():
    """
    Generic data loading class for speech emphasis detection datasets.
    
    Handles dataset preprocessing, loading from disk or HuggingFace,
    adding necessary column indices, and preparing datasets for training.
    
    Attributes:
        preprocessed_dataset_path: Root directory for preprocessed dataset local storage
        model_with_emphasis_head: Model with emphasis detection capability
        hf_token: HuggingFace API token for accessing datasets
        ds_hf_train: HuggingFace dataset name for training (if applicable and the data is also used for training the emphasis detection head)
        ds_hf_eval: HuggingFace dataset name for evaluation (if applicable and the data is also used for evaluation of the emphasis detection head)
        emphasis_indices_column_name: Column name for emphasis labels
        columns_to_remove: Columns to exclude from the dataset
        split_train_val_percentage: Percentage of data to use for validation
    """
    
    def __init__(self, 
                preprocessed_dataset_path, 
                columns_to_remove, 
                model_with_emphasis_head, 
                hf_token=None,
                ds_hf_train=None, 
                ds_hf_eval=None,
                emphasis_indices_column_name="emphasis_indices", 
                transcription_column_name='transcription', 
                split_train_val_percentage=0.02
            ):
        
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.model_with_emphasis_head = model_with_emphasis_head
        self.hf_token = hf_token
        self.ds_hf_train = ds_hf_train
        self.ds_hf_eval = ds_hf_eval
        self.emphasis_indices_column_name = emphasis_indices_column_name
        self.columns_to_remove = columns_to_remove
        self.transcription_column_name = transcription_column_name
        self.split_train_val_percentage = split_train_val_percentage
        self.dataset = self.load_preproc_datasets(model_with_emphasis_head, 
                                                preprocessed_dataset_path, 
                                                columns_to_remove,
                                                emphasis_indices_column_name, 
                                                transcription_column_name, 
                                                ds_hf_train, 
                                                hf_token)

    def load_preproc_datasets(self, 
                            model_with_emphasis_head, 
                            preprocessed_dataset_path, 
                            columns_to_remove,
                            emphasis_indices_column_name, 
                            transcription_column_name,
                            ds_name_hf, 
                            hf_token):
        """
        Load and preprocess datasets from disk or HuggingFace.
        
        If the dataset exists on disk, loads it directly. Otherwise, downloads and
        processes it using the DSProcessor, then saves it to disk for future use.
        Also adds sentence indices and performs necessary column transformations.
        
        Args:
            model_with_emphasis_head: Model with emphasis detection capability
            columns_to_remove: Columns to exclude from the dataset
            emphasis_indices_column_name: Column name for emphasis labels
            transcription_column_name: Column name for transcription text
            ds_name_hf: HuggingFace dataset name
            hf_token: HuggingFace API token
            
        Returns:
            Processed dataset with unnecessary columns removed
        """
        def change_input_features(example):
            example['input_features'] = example['input_features'][0]
            return example
        def add_sentence_index(row, index_container):
            curr_index = index_container['sentence_index']
            row['sentence_index'] = curr_index
            index_container["sentence_index"] += 1
            return row
        
        if os.path.exists(preprocessed_dataset_path):
            train_set = load_from_disk(preprocessed_dataset_path)
            return train_set.remove_columns(columns_to_remove)
        else:
            ds_preprocessor = DSProcessor(
                ds_name=ds_name_hf,
                processor=model_with_emphasis_head.processor,
                hyperparameters={"split_train_val_percentage": self.split_train_val_percentage},
                hf_token=hf_token
            )
            train_set = ds_preprocessor.get_train_dataset(emphasis_indices_column_name=emphasis_indices_column_name, 
                                                            transcription_column_name=transcription_column_name,
                                                            model=model_with_emphasis_head,
                                                            columns_to_remove=[])
            index_container = {"sentence_index": 0}
            train_set = train_set.map(add_sentence_index, num_proc=1, load_from_cache_file=False, fn_kwargs={'index_container': index_container})
            train_set = train_set.map(change_input_features, load_from_cache_file=False, num_proc=1)
            if "labels" in train_set['train'].column_names:
                train_set = train_set.rename_column("labels", f"labels_{transcription_column_name}")
            train_set.save_to_disk(os.path.join(preprocessed_dataset_path))            
            return train_set.remove_columns(columns_to_remove)
    
    def split_train_val(self):
        """
        Split dataset into training and validation sets.
        
        Args:
            rows_to_remove: Optional list of row indices to exclude
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset, None
        dataset_split = self.dataset["train"].train_test_split(
            test_size=self.split_train_val_percentage,
            shuffle=True,
            seed=42,
        )
        return dataset_split["train"], dataset_split["test"]

class PreprocessedITAStressLoader(PreprocessedDataLoader):
    """
    Data loader for ITA stress dataset (real speech).
    """

    def __init__(
        self,
        model_with_emphasis_head,
        transcription_column_name,
        save_path,
        ita_audio_dir,
        word_dir,
        train_label_path,
        test_label_path,
    ):
        self.ita_audio_dir = Path(ita_audio_dir)
        self.word_dir = Path(word_dir)
        self.train_label_path = Path(train_label_path)
        self.test_label_path = Path(test_label_path)

        columns_to_remove = ["audio", "emphasis_indices"]

        super().__init__(
            preprocessed_dataset_path=save_path,
            columns_to_remove=columns_to_remove,
            model_with_emphasis_head=model_with_emphasis_head,
            emphasis_indices_column_name="emphasis_indices",
            transcription_column_name=transcription_column_name,
            ds_hf_train=None,   # IMPORTANT: not HF
        )

    def _load_words(self, mat_path):
        mat = sio.loadmat(mat_path, squeeze_me=True)
        for key in ["words", "word", "wrd", "transcript"]:
            if key in mat:
                return list(mat[key])
        raise RuntimeError(f"Cannot read words from {mat_path}")

    def _load_labels(self, path):
        labels = {}
        with open(path) as f:
            for line in f:
                utt, rest = line.strip().split(None, 1)
                labels[utt] = eval(rest)
        return labels

    def _build_dataset(self, split):
        if split == "train":
            audio_dir = self.ita_audio_dir / "train"
            labels = self._load_labels(self.train_label_path)
        else:
            audio_dir = self.ita_audio_dir / "test"
            labels = self._load_labels(self.test_label_path)

        rows = []
        for utt_id, stress in labels.items():
            wav = audio_dir / f"{utt_id}.wav"
            mat = self.word_dir / f"{utt_id}.mat"

            if not (wav.exists() and mat.exists()):
                continue

            words = self._load_words(mat)

            # hard safety check
            if len(words) != len(stress):
                continue

            rows.append({
                "audio": str(wav),
                "transcription": " ".join(map(str, words)),
                "emphasis_indices": stress,
                "sentence_index": 0,
            })

        ds = Dataset.from_list(rows)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds
    
        return dataset.map(
            add_idx,
            with_indices=True,
            load_from_cache_file=False
        )


    def load_preproc_datasets(
        self,
        model_with_emphasis_head,
        preprocessed_dataset_path,
        columns_to_remove,
        emphasis_indices_column_name,
        transcription_column_name,
        ds_name_hf,
        hf_token,
    ):
        if os.path.exists(preprocessed_dataset_path):
            dataset = load_from_disk(preprocessed_dataset_path)
            return dataset.remove_columns(columns_to_remove)

        # Build raw datasets
        train_ds = self._build_dataset("train")
        test_ds = self._build_dataset("test")

        raw = {"train": train_ds, "test": test_ds}

        ds_preprocessor = DSProcessor(
            ds_name=raw,
            processor=model_with_emphasis_head.processor,
            hyperparameters={"split_train_val_percentage": self.split_train_val_percentage},
            hf_token=None,
        )

        dataset = ds_preprocessor.get_train_dataset(
            emphasis_indices_column_name=emphasis_indices_column_name,
            transcription_column_name=transcription_column_name,
            model=model_with_emphasis_head,
            columns_to_remove=[],
        )

        dataset.save_to_disk(preprocessed_dataset_path)
        return dataset.remove_columns(columns_to_remove)

class PreprocessedGERStressLoader(PreprocessedITAStressLoader):
    pass


class PreprocessedTinyStress15KLoader(PreprocessedDataLoader):
    """
    Data loader for synthetic GPT-generated data.
    
    This dataset contains TTS-generated speech from GPT-written stories with
    specifically marked emphasis. This synthetic data helps supplement real
    datasets for training emphasis detection models.
    """
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path):
        # preprocessed_dataset_path = save_path # modify this path, if exists
        ds_hf_train = "slprl/TinyStress-15K" # train and val set
        columns_to_remove = ['id', 'original_sample_index', 'ssml', 'emphasis_indices', 'metadata', 'word_start_timestamps', 'audio']
        super().__init__(preprocessed_dataset_path=save_path, 
                        columns_to_remove=columns_to_remove,
                        model_with_emphasis_head=model_with_emphasis_head, 
                        emphasis_indices_column_name='emphasis_indices',
                        transcription_column_name=transcription_column_name, 
                        ds_hf_train=ds_hf_train)

    def split_train_val_test(self):
        """
        Split synthetic GPT dataset into train, validation, and test sets.
        
        Uses predefined splits from the HuggingFace dataset, with an optional
        further split of the training set for validation.
        
        Returns:
            train_set, eval_set, test_set
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        train_set, eval_set = super().split_train_val()
        return train_set, eval_set, self.dataset["test"]
    
    
def load_data(model_with_emphasis_head, transcription_column_name, dataset_name, save_path=None):
    if dataset_name == "tinyStress-15K":
        return PreprocessedTinyStress15KLoader(
            model_with_emphasis_head,
            transcription_column_name,
            save_path=save_path,
        )

    elif dataset_name == "ITA-stress":
        base = "/sd1/jhansi/interns/somesh/whisper-asl/datasets/extracted"
        return PreprocessedITAStressLoader(
            model_with_emphasis_head=model_with_emphasis_head,
            transcription_column_name=transcription_column_name,
            save_path=save_path,
            ita_audio_dir=os.path.join(base, "ITA"),
            word_dir=os.path.join(base, "word"),
            train_label_path=os.path.join(base, "energy_based_dataset", "gt_train_labels.txt"),
            test_label_path=os.path.join(base, "energy_based_dataset", "test_gt_labels.txt"),
        )
    elif dataset_name == "GER-stress":
        base = "/sd1/jhansi/interns/somesh/whisper-asl/datasets/extracted"
        return PreprocessedITAStressLoader(
            model_with_emphasis_head=model_with_emphasis_head,
            transcription_column_name=transcription_column_name,
            save_path=save_path,
            ita_audio_dir=os.path.join(base, "GER"),
            word_dir=os.path.join(base, "word"),
            train_label_path=os.path.join(base, "energy_based_dataset", "gt_ger_train_labels.txt"),
            test_label_path=os.path.join(base, "energy_based_dataset", "gt_ger_test_labels.txt"),
        )
    

    else:
        raise ValueError(f"Dataset {dataset_name} is not defined in data_loader.py")
