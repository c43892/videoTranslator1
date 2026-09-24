from videotranslator.runtime_health import ActivityWatchdog, ProcessActivity


def test_busy_or_progressing_work_has_no_total_runtime_limit():
    clock = [0]
    monitor = ActivityWatchdog(clock=lambda: clock[0])
    for seconds in range(0, 12 * 3600, 60):
        clock[0] = seconds
        assert monitor.observe(('synthesize', 77), active=True)
    clock[0] += 899
    assert monitor.observe(('synthesize', 77))
    clock[0] += 1
    assert not monitor.observe(('synthesize', 77))
    assert monitor.observe(('synthesize', 78))


def test_process_activity_does_not_count_polling_noise(tmp_path):
    process = tmp_path / '42'
    process.mkdir()
    def write(cpu, io):
        fields = ['0'] * 20
        fields[11], fields[19] = str(cpu), '123'
        (process / 'stat').write_text('42 (worker name) ' + ' '.join(fields))
        (process / 'io').write_text(f'rchar: {io}\nwchar: 0\n')
    monitor = ProcessActivity(42, tmp_path)
    write(100, 1000)
    assert not monitor.sample()
    write(100, 2000)
    assert not monitor.sample()
    write(100 + monitor.ticks, 2000)
    assert monitor.sample()
    write(100 + monitor.ticks, 100000)
    assert monitor.sample()
