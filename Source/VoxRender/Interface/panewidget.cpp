/* ===========================================================================

    Project: VoxRender
    
	Description: Implements an accordion style pane widgeth

    Copyright (C) 2013 Lucas Sherman

    MODIFIED FROM LUXRENDER'S 'panewidget.cpp'

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Include Headers
#include "ui_panewidget.h"
#include "panewidget.h"

#include <iostream>

using namespace std;

ClickableLabel::ClickableLabel(const QString& label, QWidget *parent) : QLabel(label,parent) {
}

void ClickableLabel::mouseReleaseEvent(QMouseEvent* event) 
{
	emit clicked();
}

PaneWidget::PaneWidget(QWidget *parent, const QString& label, const QString& icon, bool onoffbutton, bool solobutton) : 
    QWidget(parent), 
    ui(new Ui::PaneWidget)
{
	expanded = false;
	onofflabel = NULL;
	sololabel = NULL;

	m_Index = -1;

	ui->setupUi(this);
	
	ui->frame->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color:          \
        qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgb(120, 120, 120),  \
        stop:0.8 rgb(230, 230, 230))\n""}\n"""));

	if (!icon.isEmpty())
		ui->labelPaneIcon->setPixmap(QPixmap(icon));
		ui->labelPaneIcon->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color: rgba(232, 232, 232, 0)\n""}"));
	
	if (!label.isEmpty())
		ui->labelPaneName->setText(label);
		ui->labelPaneName->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color: rgba(232, 232, 232, 0)\n""}"));

    expandlabel.reset(new ClickableLabel(">", this));
	expandlabel->setPixmap(QPixmap(":/icons/collapsedicon.png"));
	expandlabel->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color: rgba(232, 232, 232, 0)\n""}"));
	ui->gridLayout->addWidget(expandlabel.get(), 0, 3, 1, 1);
 
	connect(expandlabel.get(), SIGNAL(clicked()), this, SLOT(expandClicked()));

	powerON = false;
	m_SoloState = SOLO_OFF;
	
	if (onoffbutton) showOnOffButton();

	if (solobutton) showSoloButton();
}

// --------------------------------------------------------------------
// Sets the pane window title
// --------------------------------------------------------------------
void PaneWidget::setTitle(const QString& title)
{
	ui->labelPaneName->setText(title);
}

// --------------------------------------------------------------------
// Sets the pane window icon
// --------------------------------------------------------------------
void PaneWidget::setIcon(const QString& icon)
{
	ui->labelPaneIcon->setPixmap(QPixmap(icon));
}

// --------------------------------------------------------------------
// Enables an on/off button for this pane
// --------------------------------------------------------------------
void PaneWidget::showOnOffButton(bool showbutton)
{
	if (onofflabel == nullptr) 
    {
		onofflabel.reset(new ClickableLabel("*", this));
		onofflabel->setPixmap(QPixmap(":/icons/poweronicon.png"));
		onofflabel->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color: rgba(232, 232, 232, 0)\n""}"));
		onofflabel->setToolTip( "Click to toggle this light on and off." );

		ui->gridLayout->removeWidget(expandlabel.get());
		ui->gridLayout->addWidget(onofflabel.get(), 0, 3, 1, 1);
		ui->gridLayout->addWidget(expandlabel.get(), 0, 4, 1, 1);

		connect(onofflabel.get(), SIGNAL(clicked()), this, SLOT(onoffClicked()));
		powerON = true;
	}

	if (showbutton) onofflabel->show();
	else            onofflabel->hide();
}

// --------------------------------------------------------------------
// Triggers the on/off state of the pane 
// --------------------------------------------------------------------
void PaneWidget::onoffClicked()
{
	if (mainwidget->isEnabled()) 
    {
		mainwidget->setEnabled(false);
		onofflabel->setPixmap(QPixmap(":/icons/powerofficon.png"));
		emit turnedOff();
		powerON = false;
	}
	else 
    {
		mainwidget->setEnabled(true);
		onofflabel->setPixmap(QPixmap(":/icons/poweronicon.png"));
		emit turnedOn();
		powerON = true;
	}
}

// --------------------------------------------------------------------
//  Toggle the panes state between expanded and collapsed
// --------------------------------------------------------------------
void PaneWidget::expandClicked()
{
	if (expanded) collapse();
	else expand();
}

// --------------------------------------------------------------------
//  Toggles the display of the solo button 
// --------------------------------------------------------------------
void PaneWidget::showSoloButton(bool showbutton)
{
	if (sololabel == nullptr) 
    {
		sololabel.reset(new ClickableLabel("S", this));
		sololabel->setPixmap(QPixmap(":/icons/lightdeleteicon.png"));
		sololabel->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color: rgba(232, 232, 232, 0)\n""}"));

		ui->gridLayout->removeWidget(expandlabel.get());
		ui->gridLayout->addWidget(sololabel.get(), 0, 3, 1, 1);
		ui->gridLayout->addWidget(onofflabel.get(), 0, 4, 1, 1);
		ui->gridLayout->addWidget(expandlabel.get(), 0, 5, 1, 1);

		connect(sololabel.get(), SIGNAL(clicked()), this, SLOT(soloClicked()));
	}

	if (showbutton) sololabel->show();
	else            sololabel->hide();
}

// --------------------------------------------------------------------
//  Signals that the solo label on the pane was clicked
// --------------------------------------------------------------------
void PaneWidget::soloClicked()
{
	if (m_SoloState == SOLO_ENABLED)
	{
		emit signalLightGroupSolo(-1);
	}
	else 
	{
		emit signalLightGroupSolo(m_Index);
	}

	emit valuesChanged();
}

void PaneWidget::SetSolo( SoloState esolo )
{
	m_SoloState = esolo;
	
	if ( m_SoloState == SOLO_ENABLED || m_SoloState == SOLO_OFF )
	{
		sololabel->setPixmap(QPixmap(":/icons/plusicon.png"));
	}
	else
	{
		sololabel->setPixmap(QPixmap(":/icons/minusicon.png"));
	}
}

// --------------------------------------------------------------------
//  Toggles display of the child widget inside the pane to shown
// --------------------------------------------------------------------
void PaneWidget::expand( )
{
	expanded = true;
	expandlabel->setPixmap(QPixmap(":/icons/expandedicon.png"));
	mainwidget->show( );
}

// --------------------------------------------------------------------
//  Toggles display of the child widget inside the pane to hidden
// --------------------------------------------------------------------
void PaneWidget::collapse( )
{
	expanded = false;
	expandlabel->setPixmap(QPixmap(":/icons/collapsedicon.png"));
	mainwidget->hide();
}

// --------------------------------------------------------------------
//  Sets the child widget for this pane
// --------------------------------------------------------------------
void PaneWidget::setWidget(QWidget *widget)
{
	mainwidget = widget;
	ui->paneLayout->addWidget(widget);
#if defined(__APPLE__)
	expandlabel->setStyleSheet(QString::fromUtf8(" QFrame {\n""background-color: rgba(232, 232, 232, 0)\n""}"));
#endif
	if (!mainwidget->isEnabled())
		onofflabel->setPixmap(QPixmap(":/icons/powerofficon.png"));
	if (expanded)
		mainwidget->show();
	else
		mainwidget->hide();
}

// --------------------------------------------------------------------
//  Returns the child widget for this pane
// --------------------------------------------------------------------
QWidget *PaneWidget::getWidget()
{
	return mainwidget;
}
