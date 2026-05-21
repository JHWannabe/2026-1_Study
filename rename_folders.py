import os
import re

base_path = r"D:\영상제공\강남\강남_원본"

folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

to_rename = []
for name in folders:
    first_part = name.split('_')[0]
    if re.fullmatch(r'\d+', first_part) and len(first_part) < 4:
        padded = first_part.zfill(4)
        new_name = name.replace(first_part, padded, 1)
        to_rename.append((name, new_name))

print(f"변경 대상 폴더 수: {len(to_rename)}")
for old, new in to_rename[:10]:
    print(f"  {old} → {new}")
if len(to_rename) > 10:
    print(f"  ... (총 {len(to_rename)}개)")

confirm = input("\n진행하시겠습니까? (y/n): ").strip().lower()
if confirm != 'y':
    print("취소되었습니다.")
    exit()

errors = []
for old, new in to_rename:
    old_path = os.path.join(base_path, old)
    new_path = os.path.join(base_path, new)
    try:
        os.rename(old_path, new_path)
    except Exception as e:
        errors.append((old, new, str(e)))

print(f"\n완료: {len(to_rename) - len(errors)}개 성공, {len(errors)}개 실패")
if errors:
    for old, new, err in errors:
        print(f"  실패: {old} → {new} | {err}")
