import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import httpx

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from videotranslator.payment_environment import select_environment, bind_store
from videotranslator.config import settings_from_env
from videotranslator.docstore import SQLiteStore
from videotranslator.domain.models import Payment, User
from videotranslator.api.app import create_app


@pytest.fixture
def profiles(tmp_path):
    result = {'PAYMENT_MODE':'live', 'PUBLIC_APP_URL':'http://localhost:8090'}
    for mode, key in [('sandbox','sk_test_example'), ('live','rk_live_example')]:
        result.update({f'STRIPE_{mode.upper()}_SECRET_KEY':key,
                       f'STRIPE_{mode.upper()}_WEBHOOK_SECRET':f'whsec_{mode}',
                       f'PAYMENT_{mode.upper()}_STORE_PATH':str(tmp_path / f'{mode}.db'),
                       f'PAYMENT_{mode.upper()}_QUEUE_PATH':str(tmp_path / f'{mode}-queue.db')})
    return result


def helper():
    spec = importlib.util.spec_from_file_location('stripe_mode', Path(__file__).resolve().parents[2]/'deploy/stripe-mode.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mode_alone_selects_credentials_store_and_queue(profiles, monkeypatch):
    for k,v in profiles.items(): monkeypatch.setenv(k,v)
    for mode in ('sandbox','live'):
        monkeypatch.setenv('PAYMENT_MODE',mode)
        selected = select_environment({**profiles,'PAYMENT_MODE':mode})
        settings = settings_from_env()
        assert selected.key == profiles[f'STRIPE_{mode.upper()}_SECRET_KEY']
        assert selected.webhook_secret == f'whsec_{mode}'
        assert settings.store_path == selected.store_path
        assert settings.local_queue_db == selected.queue_path
        assert selected.key not in repr(selected) and selected.webhook_secret not in repr(selected)


def test_active_job_limit_can_follow_deployed_provider_slots(monkeypatch):
    monkeypatch.setenv('MAX_ACTIVE_JOBS_PER_USER', '2')
    assert settings_from_env().max_active_jobs_per_user == 2


@pytest.mark.parametrize('field,value', [('STRIPE_LIVE_SECRET_KEY','sk_test_wrong'),
    ('STRIPE_LIVE_SECRET_KEY',''), ('STRIPE_LIVE_WEBHOOK_SECRET','')])
def test_missing_profile_never_falls_back_to_legacy_key(profiles, field, value):
    profiles.update({field:value,'STRIPE_SECRET_KEY':'rk_live_legacy','STRIPE_WEBHOOK_SECRET':'whsec_legacy'})
    with pytest.raises(ValueError): select_environment(profiles)


@pytest.mark.parametrize('kind',['STORE_PATH','QUEUE_PATH'])
def test_shared_or_memory_payment_files_rejected(profiles, kind):
    for path in [profiles[f'PAYMENT_SANDBOX_{kind}'],':memory:','']:
        with pytest.raises(ValueError):
            select_environment({**profiles,f'PAYMENT_LIVE_{kind}':path})


def test_existing_history_and_wallet_return_when_switching_back(profiles):
    for mode, balance in [('sandbox',1100),('live',100)]:
        store=SQLiteStore(select_environment(profiles,mode).store_path)
        bind_store(store,mode)
        with store.transaction() as tx: tx.insert(User(user_id='same-user',point_balance_units=balance),'same-user')
    for mode,balance in [('live',100),('sandbox',1100),('live',100)]:
        store=SQLiteStore(select_environment(profiles,mode).store_path)
        bind_store(store,mode)
        with store.transaction() as tx: assert tx.get(User,'same-user').point_balance_units==balance
        with pytest.raises(ValueError, match='another environment'): bind_store(store,'live' if mode=='sandbox' else 'sandbox')


def test_preexisting_sandbox_payments_cannot_be_adopted_as_live(tmp_path):
    store=SQLiteStore(str(tmp_path/'old.db'))
    with store.transaction() as tx: tx.insert(Payment(payment_id='p1',provider='stripe',provider_order_id='cs_test_old'),'p1')
    with pytest.raises(ValueError, match='Existing Stripe'): bind_store(store,'live')
    bind_store(store,'sandbox')


def test_stale_browser_cannot_create_checkout_in_new_environment(container, user):
    client=TestClient(create_app(container),headers={'Authorization':'Bearer fake:u1','X-Payment-Mode':'sandbox'})
    result=client.post('/api/v1/billing/sessions',json={'package_id':'points_1_v1','provider':'stripe'})
    assert result.status_code==409 and result.json()['detail']['code']=='payment_environment_changed'
    with container.store.transaction() as tx: assert tx.query(Payment)==[]
    assert client.get('/api/v1/me',headers={'X-Payment-Mode':'disabled'}).status_code==200


def seed_sqlite(profiles, collection=None, data=None):
    path=select_environment(profiles).store_path
    SQLiteStore(path)
    if collection:
        with sqlite3.connect(path) as conn:
            conn.execute('INSERT INTO docs VALUES (?,?,?,?)',(collection,'example',1,json.dumps(data)))


@pytest.mark.parametrize('collection,data', [('jobs',{'status':'running'}),('jobs',{'status':'queued'}),
    ('jobs',{'status':'inspecting'}),('payments',{'status':'pending'})])
def test_unfinished_work_prevents_switch_without_changing_files(profiles, collection, data):
    seed_sqlite(profiles,collection,data)
    with pytest.raises(ValueError): helper().check(profiles,'sandbox')


def test_listener_failure_leaves_original_env_intact(profiles, tmp_path, monkeypatch):
    seed_sqlite(profiles)
    env=tmp_path/'.env'
    env.write_text('\n'.join(f'{k}={v}' for k,v in profiles.items())+'\nUNRELATED=keep-me\n')
    before=env.read_bytes()
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(a,1))
    with pytest.raises(ValueError,match='authorization failed'): helper().write_mode(env,'sandbox')
    assert env.read_bytes()==before
    assert not list(tmp_path.glob('.env.stripe-prepare-*'))


def test_successful_switch_preserves_other_settings_and_secrets(profiles,tmp_path,monkeypatch):
    seed_sqlite(profiles)
    env=tmp_path/'.env';env.write_text('\n'.join(f'{k}={v}' for k,v in profiles.items())+'\nUNRELATED=keep-me\n')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(a,0))
    helper().write_mode(env,'sandbox')
    updated=dotenv_values(env)
    assert updated['PAYMENT_MODE']=='sandbox' and updated['UNRELATED']=='keep-me'
    for key,value in profiles.items():
        if key!='PAYMENT_MODE': assert updated[key]==value


def test_local_command_refuses_cloud_url(profiles):
    profiles['PUBLIC_APP_URL']='https://translator.example'
    with pytest.raises(ValueError,match='cloud deployments'): helper().check(profiles,'sandbox')


@pytest.mark.parametrize('status,paid,mode,allowed', [('expired','unpaid',True,True),
    ('open','unpaid',True,False), ('complete','paid',True,False),
    ('complete','unpaid',True,False), ('expired','unpaid',False,False)])
def test_only_verified_expired_unpaid_checkout_allows_switch(profiles,monkeypatch,status,paid,mode,allowed):
    seed_sqlite(profiles,'payments',{'payment_id':'p1','provider':'stripe','status':'pending','provider_order_id':'cs_live_saved'})
    monkeypatch.setattr(httpx,'get',lambda *a,**kw:httpx.Response(200,json={
        'id':'cs_live_saved','client_reference_id':'p1','status':status,'payment_status':paid,'livemode':mode}))
    if allowed:
        helper().check(profiles,'sandbox')
    else:
        with pytest.raises(ValueError,match='Unresolved top-up'): helper().check(profiles,'sandbox')
