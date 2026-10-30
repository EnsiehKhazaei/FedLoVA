# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 17:27:08 2025

@author: ensie
"""
import numpy as np
from collections import defaultdict
from federatedscope.core.splitters import BaseSplitter
from collections import Counter

class DomainSplitter(BaseSplitter):
    """
    This splitter assigns one domain to each client, in order.

    Example:
        - Ignore the 1st domain
        - If there are n clients, select domains [2..n+1]
        - Assign domain 2 -> client 1, domain 3 -> client 2, ..., domain (n+1) -> client n

    Args:
        client_num: number of clients to split into
    """
    def __init__(self, client_num):
        super(DomainSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None):
        from torch.utils.data import Dataset, Subset
        from datasets import Dataset as HuggingFaceDataset  # for HF datasets

        # assume dataset has a 'domain' field (like MultilingualSentiment)
        # get unique domains in sorted order
        # domains = sorted(set(dataset["domain"]))
        domain_counts = Counter(dataset["domain"])

        # Sort domains by size (largest first)
        domains = domain_counts.most_common()
        print("Domains sorted by size with sample counts:")
        for domain, count in domains:
            print(f"Domain: {domain}, Samples: {count}")

        # skip the first domain, then take the next `client_num`
        selected_domains = ["en", "de", "zh"]#[d for d, _ in domains[1:self.client_num+1]]
        print(selected_domains)

        data_list = []
        for dom in selected_domains:
            # filter samples belonging to this domain
            idxs = [i for i, d in enumerate(dataset["language"]) if d == dom]
            
            if isinstance(dataset, Dataset) or isinstance(dataset, HuggingFaceDataset):
                data_list.append(Subset(dataset, idxs))
            else:
                data_list.append([dataset[i] for i in idxs])

        return data_list
