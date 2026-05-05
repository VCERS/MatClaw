Note that the PES models require PYG backend, while the property prediction 
models require DGL backend. However, only one backend can be used per python 
process, so the tests will fail if running all tests together via: 
`pytest tests/matcalc/ -v`.

The solution is to run each test file separately, e.g.:
`pytest tests/matcalc/test_matcalc_calc_elasticity.py -v`