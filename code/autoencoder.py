import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=256, device='cpu'):
        super(Autoencoder, self).__init__()
        # Enhanced Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.ReLU(True)
        ).to(device)

        # Enhanced Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim),
            nn.ReLU(True)
        ).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def train_model(self, train_loader, val_loader, rag_model, epoch, args):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)
        self.train()

        rag_model.eval()
        progress_bar = tqdm(train_loader, desc="Epoch {} Train AE".format(epoch))
        for batch in progress_bar:
            # get hidden states from RAG model
            input_ids, attention_mask, _ = [b.to(args.device) for b in batch]
            question_hidden_states = rag_model.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
            # train autoencoder
            optimizer.zero_grad()
            output = self.forward(question_hidden_states)
            loss = criterion(output, question_hidden_states)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)

        val_loss = self.inference(val_loader, rag_model, criterion, args)
        print(f'Epoch {epoch+1}, AE Validation Loss: {val_loss}')

    def inference(self, data_loader, rag_model, criterion, args):
        total_loss = 0.0
        self.eval()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="AE Inference"):
                # get hidden states from RAG model
                input_ids, attention_mask, labels = [b.to(args.device) for b in batch]
                question_hidden_states = rag_model.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
                output = self.forward(question_hidden_states)
                loss = criterion(output, question_hidden_states)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss
