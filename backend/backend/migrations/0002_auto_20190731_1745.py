# Generated by Django 2.2.1 on 2019-07-31 09:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sentence',
            name='id',
            field=models.IntegerField(primary_key=True, serialize=False, verbose_name='Sentence id'),
        ),
    ]
