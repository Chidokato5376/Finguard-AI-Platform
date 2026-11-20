# Cổng 1 — tổng hợp theo seed (val HI-Small)

Seed: 42. Mỗi ô: trung vị [min; max] theo seed.

## P1 — DRRE R5 so với mô hình cổ điển cùng 28 cột

- AUPRC DRRE: 0.3549 [0.3549; 0.3549]
- Bắt được trong 2.000 cảnh báo: 411.0000 [411.0000; 411.0000]
- Δ AUPRC so với hạng 1: -0.2186 [-0.2186; -0.2186]
- GBDT (một lần, không phụ thuộc seed khi không dừng sớm): {'gbdt_28': 0.573495978249219, 'gbdt_20': 0.4603124668985599}

## P2 — GBDT có thêm embedding/điểm DRRE

| Biến thể | AUPRC | Δ so với GBDT | Số seed có KTC 95% > 0 |
|---|---|---|---|
| gbdt | 0.5586 [0.5586; 0.5586] | — | 0/— |
| gbdt+diem_drre | 0.5413 [0.5413; 0.5413] | -0.0173 [-0.0173; -0.0173] | 0/1 |
| gbdt+emb_drre | 0.4917 [0.4917; 0.4917] | -0.0669 [-0.0669; -0.0669] | 0/1 |
| gbdt+emb_drre+diem | 0.4817 [0.4817; 0.4817] | -0.0769 [-0.0769; -0.0769] | 0/1 |

## P4 — recall theo kiểu hình trong 2.000 cảnh báo đầu (trung vị theo seed)

- BIPARTITE: 0.8000 [0.8000; 0.8000]
- CYCLE: 0.8235 [0.8235; 0.8235]
- FAN-IN: 0.5682 [0.5682; 0.5682]
- FAN-OUT: 1.0000 [1.0000; 1.0000]
- GATHER-SCATTER: 0.8429 [0.8429; 0.8429]
- RANDOM: 0.8182 [0.8182; 0.8182]
- SCATTER-GATHER: 0.7419 [0.7419; 0.7419]
- STACK: 0.7429 [0.7429; 0.7429]
- UNSPECIFIED: 0.1706 [0.1706; 0.1706]
