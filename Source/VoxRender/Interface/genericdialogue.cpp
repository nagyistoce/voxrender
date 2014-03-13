/* ===========================================================================

	Project: VoxRender

	Description: Generic dialogue for option set specification by user

    Copyright (C) 2014 Lucas Sherman

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
#include "genericdialogue.h"
#include "ui_genericdialogue.h"

// Include Dependencies
#include "VoxLib/Error/Error.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "utilities.h"

// QT Headers
#include <QLabel>

using namespace vox;
using namespace volt;

// --------------------------------------------------------------------
//  Initialize the light dialogue check state
// --------------------------------------------------------------------
GenericDialogue::GenericDialogue(String const& title, std::list<FilterParam> params) : 
    QDialog(nullptr), ui(new Ui::GenericDialogue), m_parameters(params)
{
    ui->setupUi(this);

    auto titleLabel = new QLabel(title.c_str());
    titleLabel->setAlignment(Qt::AlignHCenter);
    QFont font = titleLabel->font();
    font.setBold(true);
    titleLabel->setFont(font);
    ui->layout->addWidget(titleLabel, 0, 1);

    int row = 1;
    BOOST_FOREACH (auto & param, m_parameters)
    {
        ui->layout->addWidget(new QLabel(param.name.c_str()), row, 0);

        switch (param.type)
        {
        case FilterParam::Type_Int: {
            QSlider * slider  = new QSlider(Qt::Horizontal);
            QSpinBox * spinBox = new QSpinBox();
            auto range = boost::lexical_cast<Vector<int,2>>(param.range);
            connect(slider, SIGNAL(valueChanged(int)), spinBox, SLOT(setValue(int)));
            connect(spinBox, SIGNAL(valueChanged(int)), slider, SLOT(setValue(int)));
            spinBox->setRange(range[0], range[1]);
            slider->setMaximum(range[1]);
            slider->setMinimum(range[0]);
            spinBox->setValue(boost::lexical_cast<double>(param.value));
            ui->layout->addWidget(slider,  row, 1);
            ui->layout->addWidget(spinBox, row, 2);
            break; }
        case FilterParam::Type_Float: {
            QSlider * slider  = new QSlider(Qt::Horizontal);
            QDoubleSpinBox * spinBox = new QDoubleSpinBox();
            connect(slider, SIGNAL(valueChanged(int)), this, SLOT(valueChangeRedirect(int)));
            connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(valueChangeRedirect(double)));
            m_connMap.insert(std::make_pair(spinBox, slider));
            auto range = boost::lexical_cast<Vector<double,2>>(param.range);
            spinBox->setRange(range[0], range[1]);
            spinBox->setValue(boost::lexical_cast<double>(param.value));
            ui->layout->addWidget(slider,  row, 1);
            ui->layout->addWidget(spinBox, row, 2);
            break; }
        case FilterParam::Type_Radio: {
            break; }
        default: 
            throw Error(__FILE__, __LINE__, "GUI", "Unknown parameter type", Error_BadToken);
        }

        row++;
    }
}
    
// --------------------------------------------------------------------
//  Initialize the light dialogue check state
// --------------------------------------------------------------------
GenericDialogue::~GenericDialogue()
{
    delete ui;
}

// --------------------------------------------------------------------
//  Redirect for QT spinBox to slider
// --------------------------------------------------------------------
void GenericDialogue::valueChangeRedirect(double value) 
{
    auto spinBox = dynamic_cast<QDoubleSpinBox*>(sender());
    if (!spinBox) return;

    auto iter = m_connMap.find(spinBox);
    if (iter == m_connMap.end()) return;

    Utilities::forceSlToSb(iter->second, spinBox, value);
}

// --------------------------------------------------------------------
//  Redirect for QT slider to spinBox
// --------------------------------------------------------------------
void GenericDialogue::valueChangeRedirect(int value) 
{ 
    auto slider = dynamic_cast<QSlider*>(sender());
    if (!slider) return;
    
    BOOST_FOREACH (auto & entry, m_connMap)
    if (entry.second == slider)
    {
        Utilities::forceSbToSl(entry.first, slider, value);
    }

}

// --------------------------------------------------------------------
//  Generates an option set from the dialog state
// --------------------------------------------------------------------
void GenericDialogue::getOptions(vox::OptionSet & options)
{
    auto row = 1;
    BOOST_FOREACH (auto & param, m_parameters)
    {
        switch (param.type)
        {
        case FilterParam::Type_Int: {
            auto spin = dynamic_cast<QSpinBox*>(ui->layout->itemAtPosition(row, 2)->widget());
            if (!spin) return;
            options.addOption(param.name, spin->value());
            break; }
        case FilterParam::Type_Float: {
            auto spin = dynamic_cast<QDoubleSpinBox*>(ui->layout->itemAtPosition(row, 2)->widget());
            if (!spin) return;
            options.addOption(param.name, spin->value());
            break; }
        case FilterParam::Type_Radio: {
            options.addOption(param.name, param.value);
            break; }
        default: 
            throw Error(__FILE__, __LINE__, "GUI", "Unknown parameter type", Error_BadToken);
        }

        row++;
    }
}