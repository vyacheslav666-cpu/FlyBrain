# Сторонние библиотеки для страницы

Это three.js версии **0.170.0** (лицензия MIT, текст — `LICENSE-three.txt`). Файлы взяты как есть из npm-пакета `three@0.170.0` и не изменены. Здесь нет сборки и Node.js: браузер читает их напрямую (D-026).

| Файл | Откуда | SHA-256 |
|---|---|---|
| `three.module.js` | `three@0.170.0/build/three.module.js` | `ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb` |
| `OrbitControls.js` | `three@0.170.0/examples/jsm/controls/OrbitControls.js` | `80efaadea4f8a636a65fb0bd08bfef62f3d93a0bb94e2e7500f23176c5c07f4e` |
| `LICENSE-three.txt` | `three@0.170.0/LICENSE` | `4c40a1ef62450b857c3b2aaf294936304cd552d965fbcd9d32d4c5bcf4ba4454` |
| `jsm/postprocessing/UnrealBloomPass.js` | `three@0.170.0/examples/jsm/postprocessing/UnrealBloomPass.js` | `3bd23a1097af75c7002d0ffc21a6c14f45c4dd701dbaf737030dfc61fb7c64d9` |
| `jsm/postprocessing/Pass.js` | `three@0.170.0/examples/jsm/postprocessing/Pass.js` | `b3c6128340eaa37e40a6a2f1b738e894c855239417d50959759b34a2b5e89f92` |
| `jsm/shaders/CopyShader.js` | `three@0.170.0/examples/jsm/shaders/CopyShader.js` | `4e3346db194db56a596cd074e9bdb39fb5eb52040c333e0d29dc4eb1324d3b1d` |
| `jsm/shaders/LuminosityHighPassShader.js` | `three@0.170.0/examples/jsm/shaders/LuminosityHighPassShader.js` | `9f4866f9abb2d96fd83eec46ba4bf2165b22155a7a37ff425c0f60eba18007cb` |

Файлы из `jsm/` — эффект свечения (bloom) для «живого» режима (C9, D-033). Они лежат в тех же подпапках, что в пакете three.js, потому что ссылаются друг на друга относительными путями (`../shaders/CopyShader.js`). Сумма всего пакета `three-0.170.0.tgz` из npm — `4a608a355dcaba72e0e5383cdc814303f5b6060b43c238cdf6932dceb699238d`.

Суммы проверяет тест `tests/test_web.py`.
