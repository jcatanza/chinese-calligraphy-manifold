import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pygame, sys, os, math, csv
import torchvision.transforms as transforms
from PIL import Image

# --- CONFIG ---
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
W, H, LATENT, BATCH = 1200, 800, 3, 150
DEV = torch.device('cpu')
print("SYSTEM: CHINESE MNIST (MORPH ENGINE)")


# --- DRIVER ---
class ChineseMNIST(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.data = []
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.labels = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
            11: 10, 12: 11, 13: 12, 14: 13, 15: 14
        }

        csv_path = os.path.join(root_dir, "chinese_mnist.csv")
        img_dir = os.path.join(root_dir, "data")

        if not os.path.exists(csv_path):
            print(f"[ERROR] MISSING DATA. Please restore files to ./data/")
            self.missing = True
        else:
            self.missing = False
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    fname = f"input_{row[0]}_{row[1]}_{row[2]}.jpg"
                    fpath = os.path.join(img_dir, fname)
                    label = self.labels.get(int(row[2]), 0)
                    self.data.append((fpath, label))

    def __len__(self):
        return len(self.data) if not self.missing else 100

    def __getitem__(self, idx):
        if self.missing: return torch.zeros(1, 28, 28), 0
        fpath, label = self.data[idx]
        try:
            img = Image.open(fpath)
            return self.transform(img), label
        except:
            return torch.zeros(1, 28, 28), label


FULL_DS = ChineseMNIST("./data")
CLASS_NAMES = ["LING", "YI (1)", "ER (2)", "SAN (3)", "SI (4)", "WU (5)", "LIU (6)", "QI (7)", "BA (8)", "JIU (9)",
               "SHI (10)", "BAI", "QIAN", "WAN", "YI"]
COLORS = [(200, 200, 200), (255, 50, 50), (50, 255, 255), (50, 255, 50), (255, 255, 0), (50, 100, 255), (255, 50, 255),
          (180, 50, 255), (255, 150, 0), (200, 255, 100), (100, 100, 255), (100, 255, 100), (255, 100, 100),
          (0, 150, 255), (150, 0, 255)]


def get_pop(size):
    if FULL_DS.missing: return torch.zeros(size, 784).to(DEV), []
    idx = torch.randperm(len(FULL_DS))[:size]
    dat = torch.zeros((size, 784))
    meta = []
    for i, x in enumerate(idx):
        img, lbl = FULL_DS[x.item()]
        dat[i] = img.view(-1)
        meta.append((COLORS[lbl], CLASS_NAMES[lbl]))
    return dat.to(DEV), meta


# --- BRAIN ---
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 256), nn.LeakyReLU(0.2), nn.Linear(256, 512), nn.LeakyReLU(0.2),
                                 nn.Linear(512, 784), nn.Tanh())

    def forward(self, x): z = self.enc(x); return self.dec(z), z

    def decode(self, z): return self.dec(z)


def rot(p, ax, ay):
    x, y, z = p
    xz = complex(x, z) * complex(math.cos(ay), math.sin(ay));
    x, z = xz.real, xz.imag
    yz = complex(y, z) * complex(math.cos(ax), math.sin(ax));
    y, z = yz.real, yz.imag
    return [x, y, z]


# --- MAIN ---
def run():
    print(">> STARTING MORPH ENGINE...")
    pygame.init()
    scr = pygame.display.set_mode((W, H))
    clk = pygame.time.Clock()
    fnt = pygame.font.SysFont("Courier New", 14, bold=True)
    big = pygame.font.SysFont("Impact", 24)

    model = AE().to(DEV)
    opt = optim.Adam(model.parameters(), lr=0.002)
    crit = nn.MSELoss()

    pop, meta = get_pop(BATCH)
    cols = [m[0] for m in meta] if meta else [(255, 255, 255)] * BATCH
    names = [m[1] for m in meta] if meta else ["ERR"] * BATCH

    zoom, frame, tspread, ax, ay = 50.0, 0, 4.0, 0.0, 0.0
    s_id, e_id, prog, h_id = -1, -1, 0.0, -1

    run = True
    while run:
        mp = pygame.mouse.get_pos()
        btns = pygame.mouse.get_pressed()
        for e in pygame.event.get():
            if e.type == pygame.QUIT: run = False

            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:  # Resample
                    pop, meta = get_pop(BATCH)
                    if meta: cols, names = [m[0] for m in meta], [m[1] for m in meta]
                    s_id, e_id = -1, -1
                if e.key == pygame.K_SPACE:  # Reset
                    s_id, e_id = -1, -1

            if e.type == pygame.MOUSEBUTTONDOWN:
                if h_id != -1:
                    if e.button == 1:
                        s_id = h_id
                    elif e.button == 3:
                        e_id = h_id; prog = 0.0

        if btns[0] and h_id == -1:
            rel = pygame.mouse.get_rel()
            ay += rel[0] * 0.01;
            ax += rel[1] * 0.01
        else:
            pygame.mouse.get_rel()

        # Physics
        if frame > 0 and frame % 1000 == 0:
            for pg in opt.param_groups: pg['lr'] = max(pg['lr'] * 0.9, 0.0001)

        opt.zero_grad()
        rec, lat = model(pop)
        var_loss = torch.nn.functional.relu(tspread - torch.var(lat, dim=0).sum())
        loss = crit(rec, pop) * 1000.0 + torch.mean(lat ** 2) * 0.01 + var_loss * 0.5
        loss.backward();
        opt.step()

        with torch.no_grad():
            crd = lat.detach().numpy()
        md = np.max(np.abs(crd))
        zoom += ((350.0 / (md + 1.0) * 0.8) - zoom) * 0.1
        zoom = max(5.0, min(zoom, 150.0))

        # Render
        pts = []
        cx, cy = 600, 400
        scr.fill((10, 15, 20))

        for i in range(BATCH):
            rx, ry, rz = rot(crd[i], ax, ay)
            pts.append(
                {'i': i, 'x': rx, 'y': ry, 'z': rz, 'sx': cx + int(rx * zoom), 'sy': cy - int(ry * zoom), 'c': cols[i]})
        pts.sort(key=lambda p: p['z'])

        h_id = -1
        # Draw Line
        if s_id != -1 and e_id != -1:
            try:
                p1 = next(p for p in pts if p['i'] == s_id)
                p2 = next(p for p in pts if p['i'] == e_id)
                pygame.draw.line(scr, (255, 255, 255), (p1['sx'], p1['sy']), (p2['sx'], p2['sy']), 2)
            except:
                pass

        for p in pts:
            rad = int(max(3, 6 + p['z']))
            d = np.sqrt((p['sx'] - mp[0]) ** 2 + (p['sy'] - mp[1]) ** 2)
            if d < 10: h_id = p['i']

            sel = (p['i'] == s_id or p['i'] == e_id)
            if d < 10 or sel:
                hc = (50, 255, 50) if p['i'] == s_id else ((255, 50, 50) if p['i'] == e_id else (255, 255, 255))
                pygame.draw.circle(scr, hc, (p['sx'], p['sy']), rad + 4, 2)

            c = p['c']
            df = max(0.3, min(1.2, 1.0 + p['z'] * 0.2))
            dc = (min(255, int(c[0] * df)), min(255, int(c[1] * df)), min(255, int(c[2] * df)))
            pygame.draw.circle(scr, dc, (p['sx'], p['sy']), rad)

        # MORPH PANEL
        scale = 8
        if s_id != -1 and e_id != -1:
            t = (math.sin(prog) + 1) / 2;
            prog += 0.05
            z_mix = (1 - t) * lat[s_id] + t * lat[e_id]
            with torch.no_grad():
                im = model.decode(z_mix.unsqueeze(0)).view(28, 28).numpy() * 0.5 + 0.5

            scr.blit(big.render(f"MORPH: {int(t * 100)}%", 1, (255, 255, 255)), (50, 350))
            for r in range(28):
                for c in range(28):
                    v = int(max(0, min(1, im[r, c] * 1.5)) * 255)
                    if v > 20:
                        tc = cols[s_id] if t < 0.5 else cols[e_id]
                        tc = (int(v * (tc[0] / 255)), int(v * (tc[1] / 255)), int(v * (tc[2] / 255)))
                        pygame.draw.rect(scr, tc, (50 + c * 8, 390 + r * 8, 8, 8))

        # Instructions
        scr.blit(fnt.render("L-CLICK: SET START", 1, (100, 255, 100)), (20, 20))
        scr.blit(fnt.render("R-CLICK: SET END", 1, (255, 100, 100)), (20, 40))
        scr.blit(fnt.render("SPACE:   RESET", 1, (200, 200, 200)), (20, 60))
        if s_id != -1: scr.blit(fnt.render(f"START: {names[s_id]}", 1, cols[s_id]), (20, 100))
        if e_id != -1: scr.blit(fnt.render(f"END:   {names[e_id]}", 1, cols[e_id]), (20, 120))

        scr.blit(fnt.render(f"FR: {frame}", 1, (150, 150, 150)), (750, 750))
        pygame.display.flip()
        clk.tick(60)
        frame += 1
    pygame.quit()


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"ERROR: {e}")