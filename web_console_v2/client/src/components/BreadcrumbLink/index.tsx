/* istanbul ignore file */

import React, { FC } from 'react';
import styled from 'styled-components';
import { Breadcrumb } from '@arco-design/web-react';
import Slash from './Slash';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const Container = styled(Breadcrumb)`
  padding-left: 3px;
`;

type Props = {
  paths: {
    /** Display label or i18n key */
    label: string;
    /** link */
    to?: string;
  }[];
};

const BreadcrumbLink: FC<Props> = ({ paths }) => {
  const { t } = useTranslation();

  return (
    <Container separator={<Slash />}>
      {paths.map((item, index) => (
        <Breadcrumb.Item key={item.label}>
          {index === paths.length - 1 ? t(item.label) : <Link to={item.to!}>{t(item.label)}</Link>}
        </Breadcrumb.Item>
      ))}
    </Container>
  );
};

export default BreadcrumbLink;
