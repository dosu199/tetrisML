# Plan rada — Tetris ML

Zadnje ažurirano: 5. oktobra 2026.

Ovo je radni dokument. `STATE.md` je detaljna tehnička evidencija na engleskom;
ovo je kratka verzija s konkretnim koracima.

**Cilj:** neuronska mreža koja dobro igra Tetris.

---

## Gdje smo stali

**Konvolucijska neuronska mreža (`cnn-model.json`, 2 poteza) igra jednako dobro
kao bot s pretragom.** Na 20 partija od 2.000 figura i na 10 partija od 5.000
figura: isti prosjek kao bot, **nijedna izgubljena partija**. Ima samo **2.129
parametara** — 45 puta manje od prethodne mreže koja je igrala lošije.

Put do ovdje: pretraga na dva poteza s mrežom, dva kruga DAgger-a, pa
konvolucija. Početkom oktobra mreža je bila na 26% nivoa bota.

Mreža je sada stigla do plafona svog učitelja. Dalje od toga može samo ako uči
iz vlastite igre (korak 6) — učitelj više nema šta da je nauči.

---

## Rezultati — kako smo stigli dovde

Kapa od 500 figura (maksimum 200 linija):

| mreža | pretraga | linija | % bota | umrla |
|---|---|---|---|---|
| kloniranje ponašanja | — | 3,4 | 1,7% | 20/20 |
| mreža vrijednosti (MSE) | 1 potez | 24,2 | 12,2% | 20/20 |
| ranking (RankNet) | 1 potez | 51,3 | 25,8% | 20/20 |
| ranking | 2 poteza, **s bugom** | 33,4 | 16,8% | 20/20 |
| ranking | **2 poteza, popravljeno** | **184,9** | **93,2%** | 2/20 |

Kapa od 500 je postala preniska da razlikuje, pa duže partije, kapa 2.000
(maksimum 800 linija):

| mreža | pretraga | medijan | prosjek | % bota | umrla | slaganje na vlastitim pločama |
|---|---|---|---|---|---|---|
| ranking | 2 poteza | 679,5 | 527,8 | 66,1% | 6/10 | 75,5% |
| ranking + DAgger | 2 poteza | 797,5 | 676,7 | 84,8% | 2/10 | 80,9% |
| bot (pretraga, Faza 1) | 2 poteza | 798 | 798,2 | 100% | 0/10 | — |

Isto, ali na **20 partija** (pouzdanije), seed 20000+:

| mreža | medijan | prosjek | najgora | % bota | umrla | slaganje |
|---|---|---|---|---|---|---|
| DAgger, 1. krug | 798 | 703,6 | 23 | 88,1% | 4/20 | 80,9% |
| DAgger, 2. krug (MLP) | 798 | 762,0 | 339 | 95,4% | 3/20 | 83,0% |
| **CNN** | **798** | **798,3** | **797** | **100,0%** | **0/20** | **91,2%** |
| bot | 798,5 | 798,4 | 797 | 100% | 0/20 | — |

Još duže, kapa 5.000 (maksimum 2.000 linija), 10 partija:

| | medijan | prosjek | najgora | umrla |
|---|---|---|---|---|
| **CNN** | 1998,5 | **1998,3** | 1997 | **0/10** |
| bot | 1998 | 1998,3 | 1998 | 0/10 |

Offline, na istim test-odlukama:

| mreža | parametara | svi parovi | bliski parovi | pogodi botov najbolji |
|---|---|---|---|---|
| MLP (`rank-dagger2`) | 94.977 | 95,6% | 78,9% | 96,6% |
| **CNN** | **2.129** | **98,9%** | **93,7%** | **99,1%** |

---

## Teži režimi — izmjereno

Na običnim pravilima bot i CNN uvijek prežive, pa se ne mogu porediti. Zato
`hardmode.js` oduzima dio pomoći. 10 partija po igraču, do 2.000 figura,
2 poteza unaprijed gdje postoji pogled na sljedeću figuru:

| režim | bot prosjek figura | bot umro | CNN prosjek figura | CNN umrla |
|---|---|---|---|---|
| normalno (7-bag, vidi sljedeću) | 2000 | 0/10 | 2000 | 0/10 |
| nasumični generator, vidi sljedeću | 2000 | 0/10 | 2000 | 0/10 |
| 7-bag, **ne vidi** sljedeću | 2000 | 0/10 | 1974 | 1/10 |
| **nasumični + ne vidi sljedeću** | **2000** | **0/10** | **1444** | **6/10** |
| garbage svakih 5 figura | 216 | 10/10 | 236 | 10/10 |

Šta to znači:

**Nasumični generator sam po sebi ne pomaže.** Obje strane i dalje prežive
sve, kako je literatura i predviđala.

**Bez pogleda na sljedeću figuru mreža se lomi, a bot ne.** U najtežem
režimu bot preživi svih 10 partija, a mreža umre u 6. Mreža je bila jednaka
botu samo zato što je imala pretragu na 2 poteza da joj pokrije greške, i
zato što je trenirana isključivo na 7-bag partijama. Botova ručno smišljena
funkcija se prenosi na nove uslove; naučena mreža ne. To je opet pomak
distribucije, ovaj put u **pravilima igre**, ne u pločama.

**Garbage je jedini režim gdje obje strane gube**, i tu su otprilike
izjednačene (236 prema 216 u prosjeku, ali medijan je obrnut, 214 prema 228 —
unutar šuma za 10 partija).

### Odluka za korak 6 — tvoja

- **Garbage svakih 5:** obje strane umiru, partije su kratke (brzo za
  eksperimente), i ima mjesta da mreža postane bolja od bota.
- **Nasumični + bez pogleda:** ovdje je mreža jasno slabija od bota. Cilj bi
  prvo bio da ga dostigne (DAgger u tom režimu), a tek onda da ga nadmaši —
  ali bot ovdje ne umire, pa „bolje od bota" ne bi bilo vidljivo do duže kape.

```bash
node hardmode.js --value cnn-model.json --games 10 --cap 2000
node hardmode.js --mode memoryless-no-preview --games 20
```

## Šta smo naučili — konvolucija

**Prava pretpostavka vrijedi više od parametara.** CNN ima 45 puta manje
parametara od MLP-a, a bolja je na svakoj mjeri. MLP ima zasebnu težinu za
svaku ćeliju i za gornje redove nikad nije dobio gradijent — bio je slijep
upravo tamo gdje greška postaje fatalna. CNN koristi iste male filtere na
svakoj poziciji i prosječi rezultat preko cijele ploče, pa ono što nauči o rupi
u redu 15 važi i u redu 3. To se zove *induktivna pristrasnost*.

**Provjeri backprop prije nego mu vjeruješ.** `train_cnn.py --gradcheck`
poredi svaki ručno izveden gradijent s numeričkom procjenom. Prošao je s
greškom od 6·10⁻⁸. Bez toga bi bug u backward-u izgledao kao „mreža ne uči".

**Isti model u dva jezika mora dati isti broj.** Python trenira, JavaScript
igra. `cnn-model.json` nosi pet ploča s Python ocjenama, a JS ih na učitavanju
provjerava i odbija model ako se ne slažu.

**Iskoristi strukturu problema i za brzinu.** Mreža na nekoj poziciji vidi samo
5×5 okolinu, pa redovi neba iznad stoga daju uvijek isti doprinos. Izračunati
jednom → 3,6 puta brže, isti rezultat.

## Šta smo naučili — 2-ply i DAgger

**Bug koji je izgledao kao loša ideja.** Prva verzija 2-ply pretrage je davala
mreži samo linije s *drugog* poteza, uz obrazloženje da su linije s prvog već
„ugrađene" u nižu ploču. Pogrešno: potez koji odmah očisti liniju nije dobijao
nikakvu zaslugu, pa je pretraga naučila da odgađa čišćenje, stog je rastao, i
igra je pala na trećinu. Da sam stao kod prve mjere, zaključio bih „lookahead ne
pomaže mreži" — a pomaže 3,6 puta. **Kad rezultat protivreči očekivanju, prvo
traži grešku u vlastitom kodu.**

**DAgger pomaže politici čiju distribuciju si skupio.** Podaci za DAgger su
skupljeni dok je mreža vozila *s 2-ply pretragom*. Rezultat: 2-ply igra se
dramatično popravila (66% → 85%), a 1-ply se praktično nije pomjerila (51 → 55,
unutar šuma). Logično — popravili smo ploče do kojih dolazi 2-ply igrač, ne
1-ply igrač.

**Oprez s malim uzorkom.** 6/10 → 2/10 umrlih je ohrabrujuće, ali 10 partija je
malo. Medijan jednak botovom je jači dokaz od prosjeka.

---

## Otvorene stavke

- [ ] Ugradi CNN u igru (snippet ispod)
- [ ] Je li `main.js` obrisan?
- [ ] Commitaj i pushaj sve

---

## Plan — po redu

### ✅ 1. Više podataka — urađeno

Udvostručenje podataka pomjerilo je offline mjere za ~1,3 poena, a igru nije
pomjerilo. Podaci iz botove distribucije više nisu bili usko grlo.

### ✅ 2. Pretraga na dva poteza s mrežom — urađeno

25,8% → 93,2% na kapi 500. Najveći pojedinačni skok u projektu.

### ✅ 3. DAgger, prvi krug — urađeno

66,1% → 84,8% na kapi 2.000.

### ✅ 4. Drugi krug DAgger-a — urađeno

88,1% → 95,4%, najgora partija 23 → 339 linija. Mreža je vozila 70% figura
(beta 0,3), 208.302 nova reda, ukupno 422.004. Poboljšanje je stvarno ali manje
od prvog kruga (66% → 88%) — tipičan obrazac opadajućih prinosa.

Treći krug je opcionalan: jeftin je, ali očekuj mali pomak. Komande, ako
želiš (s `rank-dagger2.json` kao vozačem, beta 0,2):

```bash
node collect_values.js --policy rank-dagger2.json --lookahead --beta 0.2 \
     --games 60 --cap 800 --seed 70000 --out data/dagger3.csv
node merge_csv.js data/combined2.csv data/dagger3.csv --out data/combined3.csv
python3 train_rank.py --data data/combined3.csv --sample 0 --steps 30000 \
     --hidden 256,128 --model rank-dagger3.json
node play_policy.js --value rank-dagger3.json --games 20 --cap 2000 --lookahead
```

### ✅ 5. Konvolucijska mreža — urađeno

100% nivoa bota, 0 izgubljenih partija na 2.000 i na 5.000 figura. Napisana u
čistom numpy-u (`train_cnn.py`), bez PyTorcha. Treniranje traje ~30 minuta:

```bash
python3 train_cnn.py --gradcheck                      # provjera backpropa
python3 train_cnn.py --data data/combined2.csv --steps 10000
node play_policy.js --value cnn-model.json --games 20 --cap 2000 --lookahead
```

### 6. Mreža uči iz vlastite igre — SLJEDEĆE · sedmice

Učitelj nestaje: `V(s) ← r + γ·max V(s')`, nagrada = očišćene linije.

Sada je to jedini put naprijed. CNN je izjednačila učitelja, pa joj kopiranje
njegovog mišljenja ne može dati više — ona je dostigla plafon onoga što
imitacija može. Da bi igrala **bolje** od bota, mora optimizovati stvarne
linije, ne botovu procjenu.

Dobra vijest: imaš najbolju moguću polaznu tačku. RL od nule je nestabilan jer
mreža u početku igra nasumično. Ti možeš krenuti od CNN-a koja već igra kao bot,
pa je samo doučavati iz vlastitih partija.

Usput: i testovi moraju postati teži. Na 5.000 figura i bot i mreža uvijek
prežive, pa se više ne razlikuju. Treba duža igra ili brža gravitacija.

---

## Kako pokrenuti mrežu u igri

U `index.html`, u funkciji `chooserFor`, zamijeni dio za mrežu ovim — dodaje
2-ply i novi model:

```js
async function chooserFor(kind) {
  if (kind === 'search') {
    return (b, c, n) => chooseMove(b, c, n, weights, { lookahead: true });
  }
  const files = {
    value: './value-model.json',
    rank: './rank-model.json',
    dagger: './rank-dagger2.json',
    cnn: './cnn-model.json',
  };
  nets[kind] ??= loadValueNet(await json(files[kind]));
  const net = nets[kind];
  return (b, c, n) => chooseValueMove(b, c, net, { next: n, lookahead: true });
}
```

I dodaj opciju u `<select id="agent">`:

```html
<option value="dagger">Ranking mreza + DAgger, 2 poteza</option>
<option value="cnn">Konvolucijska mreza (CNN), 2 poteza</option>
```

Jedna odluka traje 6 ms za MLP i oko 10–17 ms za CNN, pa igra u browseru radi
glatko. `cnn-model.json` je samo 49 KB (MLP modeli su po 2 MB).

---

## Paralelno učenje

**Karpathy, „Neural Networks: Zero to Hero"** — epizode 1, 2, 3, 4 i 6. Epizoda
6 (WaveNet) je ista ideja kao tvoja CNN — dijeljene težine i hijerarhija.

**Sutton & Barto, poglavlja 1–6** — pred korak 6.

---

## Pravilo kojeg se držati

Svaki eksperiment završi mjerenjem **onoga što te stvarno zanima**, ne zamjenske
brojke. I kad rezultat izgleda kao da je ideja loša, prvo provjeri da nije kod
loš — ovaj krug je to pokazao bolje od bilo čega do sada.
