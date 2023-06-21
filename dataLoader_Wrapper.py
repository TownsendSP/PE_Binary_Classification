from capstone import *

import pefile
import sqlite3
import Data_Acquirer as da
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class BinaryClassificationDataset(Dataset):
    def __init__(self, x86Examples, x64Examples):
        self.x86Examples = x86Examples
        self.x64Examples = x64Examples

    def __len__(self):
        return len(self.x86Examples) + len(self.x64Examples)

    def __getitem__(self, index):
        if index < len(self.x86Examples):
            x = torch.tensor(self.x86Examples[index], dtype=torch.float32)
            y = torch.tensor(1, dtype=torch.long)
        else:
            x = torch.tensor(self.x64Examples[index - len(self.x86Examples)], dtype=torch.float32)
            y = torch.tensor(0, dtype=torch.long)
        return x, y


def get_text_section(file_path):
    pe = pefile.PE(file_path)
    text_section = None
    for section in pe.sections:
        if section.Name.decode().strip('\x00') == '.text':
            text_section = section
            break
    if text_section is None:
        raise ValueError('Could not find .text section')
    return list(text_section.get_data())

def split_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

# def extractChunks(file_info, chunk_length, num_chunks):
#     chunkList = []
#     # file_info is a list of tuples (id, file_name, file_size, path)
#     from tqdm import tqdm

#     while (len(chunkList) < num_chunks):
#         # randomly select a file from the list
#         id, file_name, file_size, path = random.choice(file_info)

#         try:
#             text_section = get_text_section(path)
#             # split the text section into 256 byte chunks
#             chunks = split_list(text_section, chunk_length)
#             # remove the last chunk if it's not 256 bytes long
#             if len(chunks[-1]) != chunk_length:
#                 chunks = chunks[:-1]
#             chunkList.append(chunks)
#         except ValueError:
#             print('Could not find .text section for {0}'.format(path))
#         except FileNotFoundError:
#             print('Could not find file {0}'.format(path))
#         except Exception as e:
#             print(e)
#             print('Could not process file {0}'.format(path))
#     return chunkList

def dataGetter(dataThing, numChunks=100, chunkLength=1000):
    num_chunks = numChunks
    chunk_length = chunkLength
    x86rows = dataThing.filter('binaries', 'platform', 'x86')
    x64rows = dataThing.filter('binaries', 'platform', 'x64')
    print("Getting file info")
    x86file_info = dataThing.getFileInfo(x86rows)
    x64file_info = dataThing.getFileInfo(x64rows)
    print("Got file info")
    #     randomly select n files from the list
    x86chunkList = extractChunks(x86file_info, chunk_length, num_chunks)
    # print the first 10 chunks, formatted as hex
    x64chunkList = extractChunks(x64file_info, chunk_length, num_chunks)
    print("Extracted chunks")
    print(f'x86 chunks: {len(x86chunkList)}')
    print(f'x64 chunks: {len(x64chunkList)}')
    return x86chunkList, x64chunkList
def extractChunks(file_info, chunk_length, num_chunks):
    chunkList = []
    # file_info is a list of tuples (id, file_name, file_size, path)
    from tqdm import tqdm

    while (len(chunkList) < num_chunks):
        # randomly select a file from the list
        id, file_name, file_size, path = random.choice(file_info)
        try:
            text_section = get_text_section(path)
            if len(text_section) < chunk_length:
                continue
            # split the text section into 256 byte chunks
            chunks = split_list(text_section, chunk_length)
            for chunk in chunks:
                if len(chunk) != chunk_length:
                    print(len(chunk))
                else:
                    chunkList.append(chunk)
        except ValueError:
            print('Could not find .text section for {0}'.format(path))
        except FileNotFoundError:
            print('Could not find file {0}'.format(path))
        except Exception as e:
            print(e)
            print('Could not process file {0}'.format(path))
        #
        # try:
        #     text_section = get_text_section(path)
        #     # split the text section into 256 byte chunks
        #     chunks = split_list(text_section, chunk_length)
        #     print(f"length of chunks: {len(chunks)}")
        #     print(f"length of chunks[-1]: {len(chunks[-1])}")
        #     print(f"datatype of first element of chunks: {type(chunks[0])}")
        #
        #     # remove the last chunk if it's not 256 bytes long
        #     if len(chunks[-1]) != chunk_length:
        #         chunks = chunks[:-1]
        #
        #
        #     chunkList.append(chunks)
        #     pbar.update(1)
        # except ValueError:
        #     print('Could not find .text section for {0}'.format(path))
        # except FileNotFoundError:
        #     print('Could not find file {0}'.format(path))
        # except Exception as e:
        #     print(e)
        #     print('Could not process file {0}'.format(path))

    return chunkList


def main():
    dataThing = da.DataAcquirer('H:/Datasets/Bigger_dataset/data.sqlite', 'H:/Datasets/Bigger_dataset/dataset/')
    dataThing.bootup()
    x86rows = dataThing.filter('binaries', 'platform', 'x86')
    x64rows = dataThing.filter('binaries', 'platform', 'x64')
    print("Getting file info")
    x86file_info = dataThing.getFileInfo(x86rows)
    x64file_info = dataThing.getFileInfo(x64rows)
    print("Got file info")
#     randomly select n files from the list
    num_chunks = 1000
    chunk_length = 256
    x86chunkList = extractChunks(x86file_info, chunk_length, num_chunks)
    x64chunkList = extractChunks(x64file_info, chunk_length, num_chunks)
    print("Extracted chunks")
    print(f'x86 chunks: {len(x86chunkList)}')
    print(f'x64 chunks: {len(x64chunkList)}')

    dataset = BinaryClassificationDataset(x86chunkList, x64chunkList)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

def main2():
#     gets 10 random binaries from the dataset, prints first 256 bytes of each's .text section
    dataThing = da.DataAcquirer('H:/Datasets/Bigger_dataset/data.sqlite', 'H:/Datasets/Bigger_dataset/dataset/')
    dataThing.bootup()
    x86rows = dataThing.filter('binaries', 'platform', 'x86')
    x64rows = dataThing.filter('binaries', 'platform', 'x64')
    print("Getting file info")
    x86file_info = dataThing.getFileInfo(x86rows)
    x64file_info = dataThing.getFileInfo(x64rows)
    print("Got file info")

#     select 10 random files from the list
    num_files = 10
    x86files = random.sample(x86file_info, num_files)
    x64files = random.sample(x64file_info, num_files)

    print("x86 files:")
    for id, file_name, file_size, path in x86files:
        md = Cs(CS_ARCH_X86, CS_MODE_32)
        print(f"File: {file_name} | Size: {file_size} | Path: {path}\n\t{[hex(x) for x in split_list(get_text_section(path), 256)[0]]}\n")
        for i in md.disasm(bytes(split_list(get_text_section(path), 256)[0]), 0x1000):
            print("\t\t0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))

    print("\n ========================================\nx64 files:\n")
    for id, file_name, file_size, path in x64files:
        md = Cs(CS_ARCH_X86, CS_MODE_64)
        print(f"File: {file_name} | Size: {file_size} | Path: {path}\n\t{[hex(x) for x in split_list(get_text_section(path), 256)[0]]}\n")
        for i in md.disasm(bytes(split_list(get_text_section(path), 256)[0]), 0x1000):
            print("\t\t0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))

if __name__ == '__main__':
    main2()