/* istanbul ignore file */

import React, { FC } from 'react';
import styled from 'styled-components';
import { useTranslation } from 'react-i18next';
import { FedRoles } from 'typings/auth';
import { MixinCircle, MixinFlexAlignCenter } from 'styles/mixins';
import { Crown, User } from 'components/IconPark';

const Container = styled.div`
  --background-color: rgb(var(--blue-1));
  --color: rgb(var(--blue-6));
  --badge-background: rgb(var(--blue-3));

  display: inline-flex;
  align-items: center;
  flex-wrap: nowrap;
  padding: 4px;
  padding-right: 10px;
  border-radius: 100px;
  font-size: 12px;
  line-height: 1;
  font-weight: normal;
  color: var(--color);
  background-color: var(--background-color);

  &[data-role='admin'] {
    --background-color: rgb(var(--gold-1));
    --color: rgb(var(--gold-6));
    --badge-background: rgb(var(--gold-3));
  }
`;

const RoleSymbol = styled.div`
  display: flex;
  ${MixinFlexAlignCenter()};
  ${MixinCircle(15)}
  margin-right: 4px;
  font-size: 9px;
  background-color: var(--badge-background);
`;

const ROLE_MAP = {
  [FedRoles.Admin]: {
    textKey: 'users.role_admin',
    icon: Crown,
  },
  [FedRoles.User]: {
    textKey: 'users.role_user',
    icon: User,
  },
};

const UserRoleBadge: FC<{ role: FedRoles }> = ({ role }) => {
  const { t } = useTranslation();

  const config = ROLE_MAP[role];

  return (
    <Container data-role={role.toLocaleLowerCase()}>
      <RoleSymbol>
        <config.icon />
      </RoleSymbol>
      {t(config.textKey)}
    </Container>
  );
};

export default UserRoleBadge;
